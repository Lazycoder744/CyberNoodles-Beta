# Make sure to install the dependencies from the requirements.txt!

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
import io

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
sys.path.insert(0, os.path.join(HERE, "onnx"))

import onnxruntime as ort          # The ONNX runtime

# Import the modules
import predict as P                # Model inputs
from bsdata import ScoreMaxSong    # Scoring system
import export_bsor as EB           # BSOR writer
from pathfinder import MapGeometry # Map Geometry stolen from an older project

# Constants used to train the model, please don't change!'
TOK = 50          # Tokens used per window (In this case 1 token per frame)
FRAME_LEN = 22    # Context frame, what can it see?
CODEBOOK = 512    # How many motion codes per layer?
RQ_LAYERS = 8     # How many layers?
MASK_ID = 512     # Token ID for Masked Prior? (Not decided yet)
DT = 1.0 / 90.0   # One frame of gameplay
FPS = 90.0        # Frames Per Second

def log(msg):
    print(f"[gen] {msg}", flush=True)


# ===========================================================================
# 1) Download the map!
# ===========================================================================
def resolve_bsr(bsr):
    """BeatSaver's API: https://api.beatsaver.com/maps/id/<key> returns everything we need, using this to fetch maps."""
    url = f"https://api.beatsaver.com/maps/id/{bsr}"
    req = urllib.request.Request(url, headers={"User-Agent": "bsai-early-access/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def download_map(bsr, maps_dir):
    """Download a map but also check if it is in cache."""
    data = resolve_bsr(bsr)
    v = data["versions"][0]                       # Pull the latest ver of the map
    h = v["hash"].lower()
    name = data.get("name", h)[:60]
    out = os.path.join(maps_dir, h)
    if os.path.isdir(out) and os.path.isfile(os.path.join(out, "Info.dat")):
        log(f"map already downloaded: {name} ({h[:8]}...)")
        return out, h.upper()
    log(f"downloading {name} -> {h[:8]}...")
    url = v.get("downloadURL") or f"https://r2cdn.beatsaver.com/{h}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "bsai-early-access/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        zdata = r.read()
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zdata)) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            target = os.path.join(out, member)
            if not os.path.abspath(target).startswith(os.path.abspath(out)):
                continue                        # Add a guard
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(z.read(member))
    log(f"extracted {len(os.listdir(out))} files")
    return out, h.upper()


# ===========================================================================
# 2) The ONNX Wrapper
# ===========================================================================
class Models:
    def __init__(self, onnx_dir, device="cpu"):
        prov = (["CPUExecutionProvider"] if device == "cpu" else
                ["CUDAExecutionProvider", "CPUExecutionProvider"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        load = lambda n: ort.InferenceSession(
            os.path.join(onnx_dir, n + ".onnx"), opts, providers=prov)
        self.enc = load("map_encoder")
        self.prior = load("masked_prior")
        self.res = load("residual")
        self.dec = load("tok_decode")
        try:
            self.hit = load("hit_scorer")
        except Exception:
            self.hit = None
        self.lrq = self.res.get_inputs()[0].shape[1]   # L from the graph (8)
        log(f"loaded 5 graphs (RQ layers = {self.lrq})")

    @staticmethod
    def _run(sess, **kw):
        names = [i.name for i in sess.get_inputs()]
        return sess.run(None, {n: kw[n] for n in names})

    # conditioning
    def encode_map(self, es, ni, wi, f0):
        """(B,6) stats, (B,50,31) notes, (B,50,6) walls, (B,50,22) frames ->
        cond (B,152,512), note_emb (B,50,512)"""
        cond, note_emb = self._run(self.enc, es=es, ni=ni, wi=wi, f0=f0)
        return cond, note_emb

    # masked prior
    def prior_logits(self, ids, visible, frame_times, cond):
        return self._run(self.prior, ids=ids, visible=visible,
                         frame_times=frame_times, cond=cond)[0]

    # residual heads: codes (B,8,50), logits (B,7,50,512)
    def residual_logits(self, codes):
        return self._run(self.res, codes=codes)[0]

    # decoder: codes (B,50,8), motion (B,50,21)
    def decode(self, codes):
        return self._run(self.dec, codes=codes)[0]

    # hit scorer
    def hit_logits(self, note_emb, motion):
        return self._run(self.hit, note_emb=note_emb, motion=motion)[0]

    def hit_score(self, note_emb, motion, notes_in):
        """Ranking scalar per candidate: mean P(present)*P(good result) over
        slots that actually contain a note. Higher = aims at the blocks."""
        hp = self.hit_logits(note_emb, motion)          # (1,50,7)
        present = 1.0 / (1.0 + np.exp(-hp[:, :, 0]))     # sigmoid
        good = np.clip(hp[:, :, 6] / 2.0, 0, 1)          # result 0..2
        has_note = (notes_in[0, :, 0] != 0).astype(np.float32)
        return float((present * good * has_note).sum()
                     / max(has_note.sum(), 1.0))


# ===========================================================================
# 3) Generate stuff
# ===========================================================================
def maskgit_window(M, cond, frame_times, cfg_scale, temp, iters, rng):
    B = 1
    ids = np.full((B, TOK), MASK_ID, dtype=np.int64)
    unknown = np.ones((B, TOK), dtype=bool)

    # unconditional conditioning = encoder on zeroed raw inputs (the exact
    # "no context" distribution training used, not zeros in token space)
    z_es = np.zeros_like(_LAST_ES)
    z_ni = np.zeros_like(_LAST_NI)
    z_wi = np.zeros_like(_LAST_WI)
    z_f0 = np.zeros_like(_LAST_F0)
    cond_u = M.encode_map(z_es, z_ni, z_wi, z_f0)[0] if cfg_scale > 0 else None

    for it in range(iters):
        # cosine schedule
        keep_n = max(1, int(round(np.cos(np.pi * it / iters) * TOK)))
        logits = M.prior_logits(ids, unknown, frame_times, cond)
        if cond_u is not None:
            logits_u = M.prior_logits(ids, unknown, frame_times, cond_u)
            logits = logits_u + cfg_scale * (logits - logits_u)
        probs = softmax(logits / temp, axis=-1)
        flat = probs.reshape(-1, CODEBOOK)
        pick = rng.multinomial
        # sample one code per (batch,slot) row
        sampled = np.array([rng.choice(CODEBOOK, p=flat[i]) for i in range(len(flat))],
                           dtype=np.int64).reshape(B, TOK)
        conf = np.take_along_axis(probs, sampled[..., None], -1)[..., 0]
        conf = np.where(unknown, conf, 10.0)
        n_commit = TOK - keep_n
        if n_commit > 0:
            thresh = np.partition(conf.flatten(), -n_commit)[-n_commit]
            newly = unknown & (conf >= thresh)
        else:
            newly = np.zeros_like(unknown)
        if not newly.any() and unknown.any():
            i = int(np.argmax(np.where(unknown, conf, -1)))
            newly = np.zeros_like(unknown); newly[i // TOK, i % TOK] = True
        ids = np.where(newly, sampled, ids)
        unknown &= ~newly
    return ids % CODEBOOK


# globals stashed by the caller for the CFG unconditional pass
_LAST_ES = _LAST_NI = _LAST_WI = _LAST_F0 = None


def residual_fill(M, base, rng, r_temp=0.0):
    L = M.lrq
    codes = np.zeros((1, L, TOK), dtype=np.int64)
    codes[:, 0] = base
    logits = M.residual_logits(codes)
    for j in range(1, L):
        h = logits[:, j - 1]
        if r_temp <= 0.0:
            codes[:, j] = h.argmax(-1)
        else:
            p = softmax(h / r_temp, -1)
            codes[:, j] = np.array(
                [rng.choice(CODEBOOK, p=p[0, t]) for t in range(TOK)],
                dtype=np.int64)
    return codes


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def generate_replay(args):
    rng = np.random.default_rng(args.seed)
    M = Models(args.onnx_dir, args.device)

    # Get the map
    if args.local_map:
        map_dir = args.local_map
        song_hash = EB.song_hash_of(map_dir)
    else:
        map_dir, song_hash = download_map(args.bsr, args.maps_dir)

    # load it through the same parser training used
    import torch  # ScoreMaxSong builds torch tensors for window encoding
    device = torch.device("cpu")
    song = P.ScoreMaxSong(map_dir, None, device, dtype=torch.float32,
                          mode=args.mode, difficulty=args.difficulty)
    song.all_elements = []
    for n in song.notes:
        song.all_elements.append({"kind": "note", "time": n["time"], "data": n})
    for b in song.bombs:
        song.all_elements.append({"kind": "bomb", "time": b["time"], "data": b})
    song.all_elements.sort(key=lambda e: e["time"])
    if getattr(args, "max_windows", 0):
        song.max_windows = args.max_windows
    geo = MapGeometry(map_dir, args.mode, args.difficulty)
    log(f"map loaded: {len(song.notes)} notes, {song.F} frames @90fps, "
        f"hash={song_hash[:8]}")

    # the chained window loop
    context = P.make_context(song, FPS)
    total_w = max(1, int(np.ceil((song.F - TOK) / TOK)))
    if getattr(song, "max_windows", 0):
        total_w = min(total_w, song.max_windows)
    log(f"generating {total_w} windows "
        f"({args.hit_cands} candidate(s) per window, iters={args.iters})")
    rows = []
    t0 = time.time()
    for k in range(total_w):
        anchor = float(context[-1, 0]) + DT   # first frame time of this window
        x0, x1, x2, x3 = P.encode_window(context, anchor, song, song.stats, DT)
        f0 = x0[..., :FRAME_LEN][None].astype(np.float32)
        ft = x1[:TOK][None].astype(np.float32)
        es = x1[TOK:][None].astype(np.float32)
        ni = x2[None].astype(np.float32)
        wi = x3[None].astype(np.float32)

        global _LAST_ES, _LAST_NI, _LAST_WI, _LAST_F0
        _LAST_ES, _LAST_NI, _LAST_WI, _LAST_F0 = es, ni, wi, f0
        cond, note_emb = M.encode_map(es, ni, wi, f0)

        # K candidates selection
        best_c, best_s = None, -1e30
        for _c in range(max(1, args.hit_cands)):
            base = maskgit_window(M, cond, ft, args.cfg, args.temp,
                                  args.iters, rng)
            codes = residual_fill(M, base, rng, r_temp=args.r_temp)
            motion = M.decode(np.transpose(codes, (0, 2, 1)))  # (B,50,8)->(B,50,21)
            if args.hit_cands > 1 and M.hit is not None:
                s = M.hit_score(note_emb, motion, ni)
                if s > best_s:
                    best_s, best_c = s, (codes, motion)
            else:
                best_c = (codes, motion)
                break
        codes, motion = best_c

        # normalize quaternions
        rec = motion[0].astype(np.float64)
        for a, b in ((3, 7), (10, 14), (17, 21)):
            q = rec[:, a:b]
            n = np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
            rec[:, a:b] = q / n
        times = anchor + np.arange(TOK) * DT
        blk = np.concatenate([times[:, None], rec], axis=1).astype(np.float32)
        rows.append(blk)
        context = blk
        if (k + 1) % max(1, total_w // 5) == 0:
            log(f"window {k+1}/{total_w} ({time.time()-t0:.0f}s)")

    G = np.concatenate(rows, axis=0)
    log(f"generated {G.shape[0]} frames in {time.time()-t0:.0f}s")

    # write the BSOR
    out = args.out or os.path.join(HERE, "out",
                                   f"{args.bsr or os.path.basename(map_dir)}"
                                   f"_{args.difficulty}.bsor")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    EB.write_bsor(geo, G, out, player_id="bsai", player_name="BSAI-EarlyAccess",
                  song_hash=song_hash, song_name=song.song_name,
                  mapper=song.mapper, environment=song.environment,
                  mode=args.mode, difficulty=args.difficulty)
    log(f"wrote {out}")
    return out

# All of the arguments and their descriptions
def main():
    ap = argparse.ArgumentParser(
        description="BSR map code -> download -> ONNX pipeline -> .bsor replay")
    ap.add_argument("bsr", nargs="?", help="BeatSaver map ID (e.g. 464f2)")
    ap.add_argument("--local-map", default=None,
                    help="use a local map directory instead of downloading")
    ap.add_argument("--difficulty", default="ExpertPlus")
    ap.add_argument("--mode", default="Standard")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="0 = whole song (default); N = stop after N windows")
    ap.add_argument("--iters", type=int, default=12,
                    help="MaskGIT rounds per window (quality/speed)")
    ap.add_argument("--temp", type=float, default=0.8,
                    help="sampling temperature (lower = safer, higher = wilder)")
    ap.add_argument("--cfg", type=float, default=3.0,
                    help="classifier-free guidance scale (0 = off)")
    ap.add_argument("--r-temp", type=float, default=0.0,
                    help="residual head temperature (0 = argmax)")
    ap.add_argument("--hit-cands", type=int, default=2,
                    help="candidates per window ranked by the hit scorer")
    ap.add_argument("--seed", type=int, default=72583)
    ap.add_argument("--device", default="cpu",
                    help="'cpu' or 'cuda' for onnxruntime")
    ap.add_argument("--onnx-dir", default=os.path.join(HERE, "onnx"))
    ap.add_argument("--maps-dir", default=os.path.join(HERE, "maps"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.bsr and not args.local_map:
        ap.error("give a BSR code or --local-map")
    os.makedirs(args.maps_dir, exist_ok=True)
    generate_replay(args)


if __name__ == "__main__":
    main()
