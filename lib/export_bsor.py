"""Export a synthesized pathfinder trajectory as a fully valid .bsor replay.

Fills every BSOR section: Info (song hash, player, difficulty, mode), Frames
(head + left/right hand), Notes (per-note annotation: hit time, cut details,
before/after/acc ratings and computed score), Walls, Heights, Pauses, and
ControllerOffsets. The per-note cut data mirrors what the game records, so the
file round-trips through the reference bsor library unchanged.

Usage:
    python scripts/export_bsor.py --map-dir DIR --difficulty D --out OUT.bsor
        [--player-id 345479 --player-name CyberRamen --song-hash HASH]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathfinder as pf  # noqa: E402
from bsor.Bsor import (  # noqa: E402
    Bsor, ControllerOffsets, Cut, Frame, Height, Info, Note, VRObject, Wall,
    NOTE_EVENT_GOOD, NOTE_EVENT_BAD, NOTE_EVENT_MISS, SABER_LEFT, SABER_RIGHT,
)
from bsor.Encoder import encode_int  # noqa: E402

CUT_DIR_VEC = {  # local-space cut direction vectors (Beat Saber convention)
    0: (0.0, 1.0, 0.0),          # up
    1: (0.0, -1.0, 0.0),         # down
    2: (-1.0, 0.0, 0.0),         # left
    3: (1.0, 0.0, 0.0),          # right
    4: (-0.707, 0.707, 0.0),     # up-left
    5: (0.707, 0.707, 0.0),      # up-right
    6: (-0.707, -0.707, 0.0),    # down-left
    7: (0.707, -0.707, 0.0),     # down-right
    8: (1.0, 0.0, 0.0),          # any
}


def compute_cuts(geo, frames):
    """Recompute the per-note cut data the way the scorer registers it.

    Returns per-note dicts with everything a BSOR Cut needs: saber speed and
    direction at the crossing, cut point/normal, distance to center, cut angle,
    and the before/after swing ratings.
    """
    F = frames.shape[0]
    t = frames[:, 0].astype(np.float64)
    head = frames[:, 1:4].astype(np.float64)
    hL = frames[:, 8:11].astype(np.float64)
    hR = frames[:, 15:18].astype(np.float64)
    qL = frames[:, 11:15].astype(np.float64)
    qR = frames[:, 18:22].astype(np.float64)
    qL = qL / (np.linalg.norm(qL, axis=1, keepdims=True) + 1e-12)
    qR = qR / (np.linalg.norm(qR, axis=1, keepdims=True) + 1e-12)
    z = np.zeros((F, 3)); z[:, 2] = 1.0
    tipL = hL + pf.q_rot_np(z, qL)
    tipR = hR + pf.q_rot_np(z, qR)

    N = len(geo.notes)
    # clamp note-crossing frame indices into the trajectory (late notes near
    # song end can otherwise point one past the final frame)
    r_e_safe = np.clip(geo.r_e.astype(np.int64), 1, F - 1)
    D = {
        "t": __import__("torch").from_numpy(t),
        "head": __import__("torch").from_numpy(head),
        "hiltL": __import__("torch").from_numpy(hL),
        "tipL": __import__("torch").from_numpy(tipL),
        "hiltR": __import__("torch").from_numpy(hR),
        "tipR": __import__("torch").from_numpy(tipR),
        "note_time": __import__("torch").from_numpy(geo.note_time),
        "line_index": __import__("torch").from_numpy(
            np.array([n["line_index"] for n in geo.notes], dtype=np.int64)),
        "line_layer": __import__("torch").from_numpy(
            np.array([n["line_layer"] for n in geo.notes], dtype=np.int64)),
        "ntype": __import__("torch").from_numpy(
            np.array([n["color"] for n in geo.notes], dtype=np.int64)),
        "cut_dir": __import__("torch").from_numpy(
            np.array([n["cut_dir"] for n in geo.notes], dtype=np.int64)),
        "e_idx": __import__("torch").arange(N),
        "n_idx": __import__("torch").arange(N),
        "r_e": __import__("torch").from_numpy(r_e_safe),
        "hand": __import__("torch").from_numpy(
            np.array([n["color"] for n in geo.notes], dtype=np.int64)),
        "cut_point": __import__("torch").zeros((N, 3)),
        "J": pf._calc_J(t),
        "NJS": geo.NJS, "bpm": geo.bpm, "JD": geo.JD, "height": geo.height,
    }
    scorer = pf.SaberScorer("cpu")
    cuts = pf._cut_points_np(scorer, D, tipL, tipR)
    D["cut_point"] = __import__("torch").from_numpy(cuts)
    res = scorer(D)

    nc = scorer.note_geometry(D)
    pos, _, fwd = scorer.note_orientations(
        D, nc, __import__("torch").arange(N), __import__("torch").arange(N),
        D["r_e"])
    pos = pos.numpy(); fwd = fwd.numpy()

    # swing-plane normals + per-frame segment angles (exact scorer math)
    normals = np.zeros((2, F, 3))
    segs = np.zeros((2, F))
    for h in (0, 1):
        tips = [tipL, tipR][h]
        hilt = [hL, hR][h]
        ax = tips - hilt
        ax_prev = np.concatenate([ax[:1], ax[:-1]])
        mid_prev = (hilt[:1] + ax[:1] / 1.0) * 0.0 + ((hilt + tips) / 2.0)
        mid_prev = np.concatenate([mid_prev[:1], mid_prev[:-1]])
        normals[h] = pf.v_norm_np(pf.v_cross_np(ax, mid_prev - hilt))
        segs[h] = pf.v_angle_np(ax, ax_prev)
        segs[h][0] = 0.0

    by_n = {r["n_idx"]: r for r in res.rec}
    total_with_mult = float(res.total)
    out = []
    for i in range(N):
        h = int(D["hand"][i])
        f0 = int(r_e_safe[i])
        tips = [tipL, tipR][h]
        hilt = [hL, hR][h]
        tip_p = tips[max(f0 - 1, 0)]
        tip_c = tips[f0]
        dt = max(t[f0] - t[f0 - 1], 1e-9) if f0 > 0 else 1.0
        saber_speed = float(np.linalg.norm(tip_c - tip_p) / dt)
        sdir = tip_c - tip_p
        sdir = sdir / (np.linalg.norm(sdir) + 1e-12)

        cpn = normals[h, f0]
        cpn = cpn / (np.linalg.norm(cpn) + 1e-12)
        npx = pf.v_cross_np(cpn, fwd[i])
        cut_point = cuts[i]
        cut_dist = float(np.abs(np.dot(cpn, cut_point - pos[i])))

        # required cut direction = note cut-dir vector rotated into world space
        n = geo.notes[i]
        q = geo.rot[i]
        q = q / (np.linalg.norm(q) + 1e-12)
        req = np.asarray(CUT_DIR_VEC.get(n["cut_dir"], (1.0, 0.0, 0.0)))
        req_w = pf.q_rot_np(req, q)
        req_w = req_w / (np.linalg.norm(req_w) + 1e-12)

        # signed deviation of the actual swing from the required direction,
        # projected onto the plane spanned by req_w and the cut normal
        dev = float(pf.v_angle_np(sdir, req_w))
        signed = dev * float(np.sign(np.dot(np.cross(req_w, sdir), cpn)))
        cut_angle = signed - 90.0

        rec = by_n.get(i, {})
        before = float(rec.get("before", 0.0))
        after = float(rec.get("after", 0.0))
        acc = int(rec.get("acc", 0))

        hit_time = float(t[f0]) if (1 <= f0 < F) else float(geo.note_time[i])
        time_dev = float(hit_time - geo.note_time[i])

        # a note is a miss if the tip never crossed its plane (far sentinel)
        is_miss = bool(np.linalg.norm(cut_point - pos[i]) > 5.0)

        # classify exactly like the game does: GOOD requires every check to pass,
        # contact that fails a check is a BAD cut, no contact is a MISS.
        speed_ok = saber_speed > 3.0
        dir_ok = abs(signed) < 110.0
        too_soon = time_dev < -0.1
        if is_miss:
            event_type = NOTE_EVENT_MISS
        elif speed_ok and dir_ok and not too_soon:
            event_type = NOTE_EVENT_GOOD
        else:
            event_type = NOTE_EVENT_BAD

        out.append({
            "note": geo.notes[i], "idx": i,
            "r_e": f0, "hand": h, "hit_time": hit_time,
            "spawn_time": hit_time - geo.JD / geo.NJS,
            "saber_speed": saber_speed,
            "saber_dir": sdir,
            # game convention: saber A (left) = 0, saber B (right) = 1
            "saber_type": h,
            "cut_point": cut_point,
            "cut_normal": cpn,
            "cut_dist": cut_dist,
            "cut_angle": cut_angle,
            "cut_deviation": signed,
            "time_dev": time_dev,
            "before": before, "after": after, "acc": acc,
            "speed_ok": speed_ok, "dir_ok": dir_ok, "too_soon": too_soon,
            "event_type": event_type,
        })
    return out, total_with_mult


def note_id(note):
    """scoringType*10000 + lineIndex*1000 + lineLayer*100 + color*10 + cutDir."""
    scoring_type = 3  # Normal
    return (scoring_type * 10000 + note["line_index"] * 1000
            + note["line_layer"] * 100 + note["color"] * 10
            + note["cut_dir"])


def write_bsor(geo, frames, out_path, player_id, player_name, song_hash,
               song_name, mapper, environment, mode, difficulty, timestamp=None,
               left_handed=False):
    import time as _time
    fps = geo.fps
    F = frames.shape[0]
    G = frames.astype(np.float64)

    info = Info()
    info.version = "1.34.0"
    info.gameVersion = "1.34.0"
    info.timestamp = timestamp or str(int(_time.time()))
    info.playerId = player_id
    info.playerName = player_name
    info.platform = "steam"
    info.trackingSystem = "OculusTouch"
    info.hmd = "OculusRift"
    info.controller = "OculusTouch"
    info.songHash = song_hash
    info.songName = song_name
    info.mapper = mapper
    info.difficulty = difficulty
    info.mode = mode
    info.environment = environment
    info.modifiers = ""
    info.jumpDistance = float(geo.JD)
    info.leftHanded = bool(left_handed)
    info.height = float(geo.height)
    info.startTime = 0.0
    info.failTime = 0.0
    info.speed = 1.0
    info.score = 0

    frames_out = []
    for i in range(F):
        fr = Frame()
        fr.time = float(G[i, 0])
        fr.fps = int(round(fps))
        h = VRObject()
        h.x, h.y, h.z, h.x_rot, h.y_rot, h.z_rot, h.w_rot = [float(v) for v in G[i, 1:8]]
        l = VRObject()
        l.x, l.y, l.z, l.x_rot, l.y_rot, l.z_rot, l.w_rot = [float(v) for v in G[i, 8:15]]
        r = VRObject()
        r.x, r.y, r.z, r.x_rot, r.y_rot, r.z_rot, r.w_rot = [float(v) for v in G[i, 15:22]]
        fr.head, fr.left_hand, fr.right_hand = h, l, r
        frames_out.append(fr)

    cut_data, total_with_mult = compute_cuts(geo, frames)
    notes_out = []
    total = 0
    for cd in cut_data:
        n = Note()
        n.note_id = note_id(cd["note"])
        n.event_time = cd["hit_time"]
        n.spawn_time = cd["spawn_time"]
        n.event_type = cd["event_type"]
        if cd["event_type"] == NOTE_EVENT_MISS:
            # whiff: no contact -> miss event carries no cut data and no score
            n.cut = None
            n.pre_score = n.post_score = n.acc_score = n.score = 0
            notes_out.append(n)
            continue
        c = Cut()
        c.speedOK = cd["speed_ok"]
        c.directionOk = cd["dir_ok"]
        c.saberTypeOk = True
        c.wasCutTooSoon = cd["too_soon"]
        c.saberSpeed = cd["saber_speed"]
        c.saberDirection = [float(v) for v in cd["saber_dir"]]
        c.saberType = cd["saber_type"]
        c.timeDeviation = cd["time_dev"]
        c.cutDeviation = cd["cut_deviation"]
        c.cutPoint = [float(v) for v in cd["cut_point"]]
        c.cutNormal = [float(v) for v in cd["cut_normal"]]
        c.cutDistanceToCenter = cd["cut_dist"]
        c.cutAngle = cd["cut_angle"]
        c.beforeCutRating = cd["before"]
        c.afterCutRating = cd["after"]
        n.cut = c
        if cd["event_type"] == NOTE_EVENT_BAD:
            # physical contact that fails a game check: bad cut, zero score
            n.pre_score = n.post_score = n.acc_score = 0
        else:
            # mirror the game's own note scoring from the cut data
            n.pre_score = int(round(cd["before"] * 70))
            n.post_score = int(round(cd["after"] * 30))
            n.acc_score = cd["acc"]
        n.score = n.pre_score + n.post_score + n.acc_score
        total += n.score
        notes_out.append(n)
    info.score = int(round(total_with_mult))

    walls_out = []
    for i, w in enumerate(geo.walls):
        wall = Wall()
        wall.id = i
        wall.energy = 0.0
        wall.time = float(w["t0"])
        wall.spawnTime = float(w["t0"] - geo.JD / geo.NJS)
        walls_out.append(wall)

    hgt = Height()
    hgt.height = float(geo.height)
    hgt.time = 0.0
    heights_out = [hgt]

    b = Bsor()
    b.magic_number = 0x442D3D69
    b.file_version = 1
    b.info = info
    b.frames = frames_out
    b.notes = notes_out
    b.walls = walls_out
    b.heights = heights_out
    b.pauses = []
    co = ControllerOffsets()
    co.left = VRObject(); co.right = VRObject()
    for obj in (co.left, co.right):
        obj.x = obj.y = obj.z = 0.0
        obj.x_rot = obj.y_rot = obj.z_rot = 0.0
        obj.w_rot = 1.0
    b.controller_offsets = co
    b.user_data = []

    with open(out_path, "wb") as f:
        b.write(f)
        # the reader expects a UserData section (magic 7) after ControllerOffsets
        # even when empty; the reference writer only emits it when non-empty.
        if b.user_data is not None and len(b.user_data) == 0:
            f.write(b"\x07")
            encode_int(f, 0)
    return info, cut_data, int(round(total_with_mult))


def song_hash_of(map_dir):
    for cand in ("song.egg", "song.ogg", "song.wav"):
        p = os.path.join(map_dir, cand)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return hashlib.sha1(f.read()).hexdigest().upper()
    return ""


def song_hash_from_bsr(bsr_id):
    """Query BeatSaver API for the authentic map version hash (uppercase)."""
    import urllib.request
    url = "https://api.beatsaver.com/maps/id/%s" % bsr_id
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return data["versions"][0]["hash"].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--difficulty", default="ExpertPlus")
    ap.add_argument("--mode", default="Standard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--player-id", default="345479")
    ap.add_argument("--player-name", default="CyberRamen")
    ap.add_argument("--song-hash", default=None)
    ap.add_argument("--bsr", default=None,
                    help="BeatSaver map id (e.g. 2777c); queries API for the authentic hash")
    ap.add_argument("--annotate-csv", default=None)
    args = ap.parse_args()

    geo = pf.MapGeometry(args.map_dir, args.mode, args.difficulty)
    left, right = pf.hand_notes(geo)
    lp = pf.plan_hand(geo, left)
    rp = pf.plan_hand(geo, right)
    lp["note_idx"] = left
    rp["note_idx"] = right
    frames = pf.build_frames(geo, lp, rp)

    m = pf.create_map(args.map_dir)
    song_meta = m.song_meta
    song_hash = args.song_hash or (song_hash_from_bsr(args.bsr) if args.bsr else song_hash_of(args.map_dir))
    info, cut_data, total = write_bsor(
        geo, frames, args.out, args.player_id, args.player_name, song_hash,
        song_meta.get("songName", ""), song_meta.get("mapper", ""),
        song_meta.get("environment", "DefaultEnvironment"),
        args.mode, args.difficulty)
    print("wrote", args.out)
    print("  song=%r mapper=%r hash=%s diff=%s/%s" % (
        info.songName, info.mapper, info.songHash, args.mode, args.difficulty))
    print("  player=%s (%s)  total score=%d" % (
        info.playerName, info.playerId, info.score))
    print("  frames=%d notes=%d walls=%d heights=%d" % (
        len(frames), len(cut_data), len(geo.walls), 1))
    hit = sum(1 for c in cut_data if c["event_type"] == NOTE_EVENT_GOOD and c["acc"] > 0)
    print("  hit notes: %d / %d" % (hit, len(cut_data)))
    if args.annotate_csv:
        import csv
        with open(args.annotate_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "time", "color", "cut_dir", "line_index", "line_layer",
                        "hit_time", "saber_speed", "before", "after", "acc",
                        "pre_score", "post_score", "acc_score", "score"])
            for cd in cut_data:
                n = cd["note"]
                w.writerow([cd["idx"], n["time"], n["color"], n["cut_dir"],
                            n["line_index"], n["line_layer"], cd["hit_time"],
                            round(cd["saber_speed"], 3), round(cd["before"], 4),
                            round(cd["after"], 4), cd["acc"],
                            int(round(cd["before"] * 70)),
                            int(round(cd["after"] * 30)), cd["acc"],
                            int(round(cd["before"] * 70)) + int(round(cd["after"] * 30)) + cd["acc"]])
        print("  annotated", args.annotate_csv)


if __name__ == "__main__":
    main()
