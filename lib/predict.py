"""Whole-song autoregressive rollout of the upstream generator + scoring + BSOR export.

Loads a GeneratorSmaller3 checkpoint, seeds a 50-frame context (first 50 frames of
a reference BSOR, or a neutral rest pose for map-only mode), then rolls out the model
window-by-window on a uniform frame grid. The generated trajectory is scored against
ALL map notes with the differentiable scorer and exported as a fully valid .bsor.

Usage:
    python3 predict.py --ckpt models/proto_best.pt --map-dir /mnt/games/bs-ai/maps/<hash> \
        --mode Standard --difficulty ExpertPlus [--seed replay.bsor] [--out-dir out]

Outputs (in --out-dir, default ./predict_out/<name>):
    replay.bsor        playable replay with per-note cut annotations
    trajectory.csv     full (t, head7, left7, right7) trajectory
    score.json         score, ref score, hit rate, jerk metrics, fps, params
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
V1 = REPO  # release folder is self-contained
sys.path.insert(0, REPO)
sys.path.insert(0, V1)

import pytorch_walls_onnx_fix3 as up  # noqa: E402
from bsdata import ScoreMaxSong  # noqa: E402
from window import encode_note_into, encode_wall_into  # noqa: E402
import export_bsor  # noqa: E402
from pathfinder import MapGeometry  # noqa: E402

FRAME_COUNT_INPUT = up.FRAME_COUNT_INPUT
FRAME_COUNT_OUTPUT = up.FRAME_COUNT_OUTPUT
FRAME_LENGTH = up.FRAME_LENGTH
NUM_TARGET_CHANNELS = up.NUM_TARGET_CHANNELS
NOTE_COUNT = up.NOTE_COUNT
NOTE_LENGTH = up.NOTE_LENGTH
WALL_COUNT = up.WALL_COUNT
WALL_LENGTH = up.WALL_LENGTH
EXTRA_STATS = up.EXTRA_STATS
MAX_TIME_AHEAD = 1.5
OUT_LEN = 21


def clamp_quat(q):
    """Normalize quaternions (…,4) along the last dim."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-6)


def make_context(song, fps):
    """First 50 context frames as (50, 22): col0=abs time, cols1:22 = transforms."""
    ctx = song.SF_ref[:FRAME_COUNT_INPUT].astype(np.float32).copy()
    if ctx.shape[0] < FRAME_COUNT_INPUT:
        pad = np.zeros((FRAME_COUNT_INPUT - ctx.shape[0], FRAME_LENGTH), np.float32)
        pad[:, 0] = ctx[-1, 0] + np.arange(1, len(pad) + 1) / fps
        pad[:, 3:7] = [0, 0, 0, 1]
        pad[:, 10:14] = [0, 0, 0, 1]
        pad[:, 17:21] = [0, 0, 0, 1]
        ctx = np.concatenate([ctx, pad])
    return ctx


def encode_window(context, anchor, song, stats, dt):
    """Encode one rollout window following upstream Extensions/preprocessing."""
    x0 = np.zeros((FRAME_COUNT_INPUT, FRAME_LENGTH), np.float32)
    x0[:, 0] = context[:, 0] - anchor
    x0[:, 1:] = context[:, 1:FRAME_LENGTH]

    x1 = np.zeros(FRAME_COUNT_OUTPUT + EXTRA_STATS, np.float32)
    x1[:FRAME_COUNT_OUTPUT] = np.arange(FRAME_COUNT_OUTPUT) * dt
    x1[FRAME_COUNT_OUTPUT:] = stats

    x2 = np.zeros((NOTE_COUNT, NOTE_LENGTH), np.float32)
    count = 0
    for e in song.all_elements:
        dte = e["time"] - anchor
        if MAX_TIME_AHEAD >= dte >= 0:
            encode_note_into(e["data"], x2, count, anchor, None)
            count += 1
            if count == NOTE_COUNT:
                break
        elif dte > MAX_TIME_AHEAD:
            break

    x3 = np.zeros((WALL_COUNT, WALL_LENGTH), np.float32)
    wcount = 0
    for w in song.walls:
        if w["time"] - anchor <= MAX_TIME_AHEAD and w["end"] - anchor >= 0:
            encode_wall_into(w, x3, wcount, anchor)
            wcount += 1
            if wcount == WALL_COUNT:
                break
        elif w["time"] - anchor > MAX_TIME_AHEAD:
            break

    return x0, x1, x2, x3


def roll_out(model, song, device, dt, n_frames, ctx0, stats):
    """Autoregressive rollout. Returns (F, 22) trajectory (t + transforms)."""
    model.eval()
    context = ctx0
    rows = []
    n_windows = max(0, int(np.ceil((n_frames - FRAME_COUNT_INPUT) / FRAME_COUNT_OUTPUT)))
    if getattr(song, "max_windows", 0):
        n_windows = min(n_windows, song.max_windows)

    with torch.no_grad():
        for k in range(n_windows):
            anchor = float(context[-1, 0]) + dt
            x0, x1, x2, x3 = encode_window(context, anchor, song, stats, dt)
            b = torch.from_numpy(x0).unsqueeze(0).to(device)
            ft = torch.from_numpy(x1[:FRAME_COUNT_OUTPUT]).unsqueeze(0).unsqueeze(-1).to(device)
            es = torch.from_numpy(x1[FRAME_COUNT_OUTPUT:]).unsqueeze(0).to(device)
            ni = torch.from_numpy(x2).unsqueeze(0).to(device)
            wi = torch.from_numpy(x3).unsqueeze(0).to(device)
            td = torch.zeros(1, NOTE_COUNT, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                pred = model(b, ft, es, ni, wi, td)
            y = pred.squeeze(0).float().cpu().numpy()  # (50, 21)
            times = anchor + np.arange(FRAME_COUNT_OUTPUT) * dt
            y = y.astype(np.float32)
            y[:, 3:7] = clamp_quat(y[:, 3:7])
            y[:, 10:14] = clamp_quat(y[:, 10:14])
            y[:, 17:21] = clamp_quat(y[:, 17:21])
            block = np.concatenate([times[:, None], y], axis=1)
            rows.append(block)
            context = block
            if len(rows) * FRAME_COUNT_OUTPUT >= n_frames - FRAME_COUNT_INPUT:
                break

    if not rows:
        return None
    G = np.concatenate(rows, axis=0)[: n_frames - FRAME_COUNT_INPUT]
    seed = ctx0
    G_full = np.concatenate([seed, G], axis=0)[:n_frames]
    return G_full


def jerk_metrics(G_np):
    if G_np.shape[0] < 2:
        return {"pct_jump": 0.0, "max_jump_m": 0.0, "max_saber_jump_m": 0.0}
    d = np.linalg.norm(np.diff(G_np[:, 1:4], axis=0), axis=1)
    dL = np.linalg.norm(np.diff(G_np[:, 8:11], axis=0), axis=1)
    dR = np.linalg.norm(np.diff(G_np[:, 15:18], axis=0), axis=1)
    dsaber = np.maximum(dL, dR)
    return {
        "pct_jump": round(float((d > 0.5).mean()) * 100.0, 2),
        "max_jump_m": round(float(d.max()), 3),
        "max_saber_jump_m": round(float(dsaber.max()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--mode", default="Standard")
    ap.add_argument("--difficulty", default="ExpertPlus")
    ap.add_argument("--seed", default=None, help="reference BSOR to seed the first 50 frames")
    ap.add_argument("--arch", default="smaller_3", help="model architecture name")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=float, default=90.0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--refine", action="store_true", help="apply One-Euro + velocity clamp post-filter")
    ap.add_argument("--refiner-ckpt", default=None, help="diffusion refiner checkpoint for NAR refinement")
    ap.add_argument("--refiner-cfg", type=float, default=0.0, help="CFG scale for the refiner (0 = off)")
    ap.add_argument("--refiner-steps", type=int, default=3)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    # the map dir is named by its authentic BeatSaver version hash; real replays
    # store it uppercase, and an empty hash makes viewers ask for a bsr code
    import re as _re
    _dir_hash = os.path.basename(os.path.normpath(args.map_dir))
    dir_song_hash = _dir_hash.upper() if _re.fullmatch(r"[0-9a-fA-F]{40}", _dir_hash) else None
    if args.seed is None:
        song = ScoreMaxSong(args.map_dir, None, device, dtype=torch.float32,
                            mode=args.mode, difficulty=args.difficulty,
                            song_hash=dir_song_hash)
    else:
        song = ScoreMaxSong(args.map_dir, args.seed, device, dtype=torch.float32,
                            song_hash=dir_song_hash)
        args.mode, args.difficulty = song.mode, song.difficulty
    song.max_windows = args.max_windows
    song.all_elements = []
    for n in song.notes:
        song.all_elements.append({"kind": "note", "time": n["time"], "data": n})
    for b in song.bombs:
        song.all_elements.append({"kind": "bomb", "time": b["time"], "data": b})
    song.all_elements.sort(key=lambda e: e["time"])

    model = up.build_model(args.arch).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print("loaded %s | params=%d | ckpt epoch=%s val=%s" % (
        args.ckpt, n_params, ck.get("epoch"), ck.get("val_loss")))

    dt = 1.0 / args.fps
    if song.replay_path:
        seed_diffs = np.diff(song.t[:max(2, min(500, len(song.t)))])
        med = float(np.median(seed_diffs[seed_diffs > 0])) if len(seed_diffs) else 0.0
        if 0 < med < 0.05:
            dt = med

    ctx0 = make_context(song, args.fps)
    n_frames = song.F
    if args.max_windows:
        n_frames = min(n_frames, FRAME_COUNT_INPUT + args.max_windows * FRAME_COUNT_OUTPUT)
    t0 = time.time()
    G = roll_out(model, song, device, dt, n_frames, ctx0, song.stats)
    secs = time.time() - t0
    if G is None:
        print("rollout produced nothing")
        sys.exit(1)

    # ---- post-refinement pipeline ----
    if args.refiner_ckpt:
        import refine
        refiner = refine.BidirectionalRefiner(d_model=256, n_blocks=4, n_heads=8).to(device)
        _rc = torch.load(args.refiner_ckpt, map_location=device, weights_only=False)
        refiner.load_state_dict(_rc["model_state"])
        refiner.eval()
        t1 = time.time()
        G = refine.refine_trajectory(G, song, dt, refiner, device,
                                     steps=args.refiner_steps, cfg=args.refiner_cfg)
        print("  refiner: %.1fs" % (time.time() - t1))
    if args.refine:
        import refine
        t1 = time.time()
        G = refine.filter_trajectory(G, song.fps)
        print("  filter: %.1fs" % (time.time() - t1))

    # align the time column to the fixture's frame grid
    G[:, 0] = song.t[:G.shape[0]]
    G_t = torch.tensor(G, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = song._score(G_t)
        score = float(out.smooth_total.item())
        ref = song.reference_score().item()

    esc = song.eval_summary(out)
    jerks = jerk_metrics(G)
    res = {
        "score": round(score, 1),
        "ref_score": round(ref, 1),
        "note_count": len(song.notes),
        "fps": round(song.fps, 1),
        "secs": round(secs, 2),
        "windows": int(np.ceil((n_frames - FRAME_COUNT_INPUT) / FRAME_COUNT_OUTPUT)),
        "params": n_params,
        "ckpt": args.ckpt,
        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in esc.items()},
        **jerks,
    }

    out_dir = args.out_dir or os.path.join(REPO, "predict_out", os.path.basename(args.map_dir))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(res, f, indent=2)

    # trajectory.csv
    import csv
    with open(os.path.join(out_dir, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "hx", "hy", "hz", "hqx", "hqy", "hqz", "hqw",
                    "lx", "ly", "lz", "lqx", "lqy", "lqz", "lqw",
                    "rx", "ry", "rz", "rqx", "rqy", "rqz", "rqw"])
        step = max(1, G.shape[0] // 2000)
        for i in range(0, G.shape[0], step):
            w.writerow(["%.4f" % v for v in G[i]])

    # valid BSOR
    geo = MapGeometry(args.map_dir, args.mode, args.difficulty)
    info, cut_data, total = export_bsor.write_bsor(
        geo, G, os.path.join(out_dir, "replay.bsor"),
        player_id="bs-ai", player_name="BS-AI",
        song_hash=song.song_hash, song_name=song.song_name,
        mapper=song.mapper, environment=song.environment,
        mode=args.mode, difficulty=args.difficulty)
    hit = sum(1 for c in cut_data if c["event_type"] == export_bsor.NOTE_EVENT_GOOD and c["acc"] > 0)
    # the actual game-style sum of per-note scores written into the BSOR
    note_sum = sum(int(round(c["before"] * 70)) + int(round(c["after"] * 30)) + int(c["acc"])
                   for c in cut_data if c["event_type"] == export_bsor.NOTE_EVENT_GOOD)
    res["bsor_total"] = int(total)          # pathfinder scorer total (proximity-based)
    res["bsor_note_score"] = note_sum       # real sum of the replay file's note scores
    res["bsor_hits"] = hit
    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("wrote %s" % out_dir)
    print("  score=%.1f ref=%.1f  hits=%d/%d  bsor_total=%d  note_score=%d" % (
        score, ref, hit, len(song.notes), total, note_sum))
    print("  jerk: %s" % jerks)
    print("  fps=%.2f (dt=%.5f)  rollout %.1fs" % (song.fps, dt, secs))


if __name__ == "__main__":
    main()