"""Build 50-frame context -> 50-frame target windows from parsed replays.

Replicates the proven upstream (DziugasRam/bs-replay-generator) schema:
  x0 (50, 22)  input frames : dt + head(7) + left(7) + right(7)
  x1 (56,)     extra stats : 50 output frame times + [njs/30, jd/30, height, pro, sc, od]
  x2 (50, 31)  notes       : upcoming notes within 1.5s (encoded)
  x3 (50, 6)   walls       : upcoming walls within 1.5s (encoded)
  y  (50, 21)  output frames: head(7) + left(7) + right(7)

Stride = frameCountOutput * 2 = 100 frames. Windows whose input/output
contain out-of-range values (> 3) are dropped (upstream `maxFrame > 3`).

Usage:
    python scripts/window.py [--config configs/data.yaml] [--limit N]
"""
import argparse
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

FRAME_COUNT_INPUT = 50
FRAME_COUNT_OUTPUT = 50
FRAME_LENGTH = 22      # dt + 21 transform values
FRAME_OUT_LENGTH = 21  # 21 transform values (no dt)
EXTRA_DETAILS_LENGTH = FRAME_COUNT_OUTPUT + 6
NOTE_COUNT = 50
NOTE_LENGTH = 31
WALL_COUNT = 50
WALL_LENGTH = 6

MAX_TIME_AHEAD = 1.5
NOTE_LINES_DISTANCE = 0.6

# Per-note hit target ("target as hits"): one fixed-size vector per note slot.
#   col0 present : 1 = this note was cut within the output window, else 0
#   col1 saber   : 0 = left, 1 = right
#   col2 hit_time: event_time (cut time) relative to window start, clamped to [-1,1]
#   col3-5 normal: cut_normal (approach direction) clamped to [-1,1]
#   col6 result  : 0 = MISS, 1 = BAD, 2 = GOOD
HIT_VEC = 7

CUTDIR_ANGLE = {0: -180, 1: 0, 2: -90, 3: 90, 4: -135, 5: 135, 6: -45, 7: 45, 8: 0}


def get_horizontal_position(line_index):
    return (-(4 - 1) * 0.5 + line_index) * NOTE_LINES_DISTANCE


def get_highest_jump_y(line_layer):
    return NOTE_LINES_DISTANCE * (line_layer + 1) + 0.05 * (5 - line_layer - (1 if line_layer > 1 else 0))


def read_frames(path):
    raw = np.fromfile(path, dtype=np.float32)
    n = len(raw) // 23
    frames = raw[: n * 23].reshape(n, 23)
    return frames


def load_notes(parsed_dir):
    """Load notes.csv as list of dicts: time (spawn), eventTime, eventType, cut info,
    plus the hit-target anchor (saber, cut point, cut normal, result)."""
    path = parsed_dir / "notes.csv"
    notes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                event_type = int(float(parts[3]))
                before_cut = float(parts[4])
                after_cut = float(parts[5])
                cut_dist = float(parts[6])
                time_dev = float(parts[7])
            except ValueError:
                continue
            # some replays carry garbage (Inf/NaN/huge) cut ratings; zero them
            if not all(np.isfinite(v) for v in (before_cut, after_cut, cut_dist, time_dev)) or \
               abs(before_cut) > 200 or abs(after_cut) > 200 or abs(cut_dist) > 200 or abs(time_dev) > 200:
                before_cut = after_cut = cut_dist = time_dev = 0.0
            # hit-target anchor (optional; absent in legacy/old notes.csv)
            if len(parts) >= 16:
                try:
                    saber_type = int(float(parts[8]))
                    cut_px = float(parts[9]); cut_py = float(parts[10]); cut_pz = float(parts[11])
                    cut_nx = float(parts[12]); cut_ny = float(parts[13]); cut_nz = float(parts[14])
                    result = int(float(parts[15]))
                    if not all(np.isfinite(v) for v in
                               (cut_px, cut_py, cut_pz, cut_nx, cut_ny, cut_nz)):
                        saber_type, cut_px, cut_py, cut_pz = -1, 0, 0, 0
                        cut_nx, cut_ny, cut_nz, result = 0, 0, 0, 0
                except ValueError:
                    saber_type, cut_px, cut_py, cut_pz = -1, 0, 0, 0
                    cut_nx, cut_ny, cut_nz, result = 0, 0, 0, 0
            else:
                saber_type, cut_px, cut_py, cut_pz = -1, 0, 0, 0
                cut_nx, cut_ny, cut_nz, result = 0, 0, 0, 0
            notes.append(
                {
                    "note_id": int(float(parts[0])),
                    "event_time": float(parts[1]),
                    "spawn_time": float(parts[2]),
                    "event_type": event_type,
                    "before_cut": before_cut,
                    "after_cut": after_cut,
                    "cut_dist": cut_dist,
                    "time_dev": time_dev,
                    "saber_type": saber_type,
                    "cut_point": [cut_px, cut_py, cut_pz],
                    "cut_normal": [cut_nx, cut_ny, cut_nz],
                    "result": result,
                }
            )
    return notes


def load_meta(parsed_dir):
    with open(parsed_dir / "meta.json") as f:
        return json.load(f)


def load_map(meta, map_dir):
    """Load the map difficulty dat for this replay's hash/difficulty/mode."""
    hash_ = meta.get("hash", "").lower()
    if len(hash_) < 40:
        return None
    zip_path = map_dir / f"{hash_}.zip"
    if not zip_path.exists():
        return None
    import zipfile
    import io

    with zipfile.ZipFile(zip_path) as z:
        # info.dat -> find difficulty filename
        try:
            info = json.loads(z.read("info.dat"))
        except KeyError:
            info = json.loads(z.read("Info.dat"))
        bpm = info.get("_beatsPerMinute", info.get("beatsPerMinute", 120.0))
        sets = info.get("_difficultyBeatmapSets", info.get("difficultyBeatmapSets", []))
        diff_file = None
        njs = 0.0
        jd = 0.0
        for s in sets:
            char_name = s.get("_beatmapCharacteristicName", s.get("beatmapCharacteristicName"))
            if char_name != meta.get("mode", "Standard"):
                continue
            for b in s.get("_difficultyBeatmaps", s.get("difficultyBeatmaps", [])):
                dname = b.get("_difficulty", b.get("difficulty"))
                if dname == meta.get("difficulty"):
                    diff_file = b.get("_beatmapFilename", b.get("beatmapFilename"))
                    njs = b.get("_noteJumpMovementSpeed", b.get("noteJumpMovementSpeed", 0.0))
                    jd = b.get("_noteJumpStartBeatOffset", b.get("noteJumpStartBeatOffset", 0.0))
                    break
        if not diff_file:
            return None
        diff = json.loads(z.read(diff_file))
    return build_map_objects(diff, njs, jd, bpm)


def build_map_objects(diff, njs, jd, bpm=120.0):
    """Convert a v2/v3 difficulty dat into note/wall objects with Time, X, Y, Rotation, Color."""
    beat_to_sec = lambda b: (b / bpm) * 60.0
    notes = diff.get("_notes", diff.get("notes", [])) or diff.get("colorNotes", [])
    bombs = []
    if not notes:
        notes = diff.get("colorNotes", [])
    bnotes = diff.get("_bombs", diff.get("bombs", [])) or diff.get("bombNotes", [])
    walls = diff.get("_obstacles", diff.get("obstacles", []))

    map_notes = []
    for n in notes:
        cd = n.get("_customData") or n.get("customData") or {}
        # NE fake notes are pure decoration: the engine never spawns or
        # scores them, so they must not enter the positional pairing with
        # replay events (they would shift every later note by one slot)
        if cd.get("_fake") or cd.get("fake"):
            continue
        type_ = n.get("_type", n.get("c", n.get("type", 0)))
        cut_dir = n.get("_cutDirection", n.get("d", n.get("cutDirection", 8)))
        x_raw = n.get("_lineIndex", n.get("x", 0))
        y_raw = n.get("_lineLayer", n.get("y", 0))
        # NE float _position: [x, y] in the same lineIndex/lineLayer space,
        # but fractional and allowed outside the 4x3 grid
        custom = cd.get("_position") or cd.get("position")
        if isinstance(custom, (list, tuple)) and len(custom) >= 2 \
                and all(isinstance(v, (int, float)) for v in custom[:2]):
            x_raw, y_raw = float(custom[0]), float(custom[1])
        beat = n.get("_time", n.get("b", n.get("time", 0.0)))
        if type_ == 0 or type_ == 1:
            map_notes.append(
                {
                    "kind": "note",
                    "beat": beat,
                    "color": type_,
                    "cut_dir": cut_dir,
                    "x": get_horizontal_position(x_raw),
                    "y": get_highest_jump_y(y_raw),
                    "angle_offset": cd.get("_cutDirection", 0.0),
                }
            )
        elif type_ == 3:
            bombs.append(
                {
                    "kind": "bomb",
                    "beat": beat,
                    "time": beat_to_sec(beat),
                    "x": get_horizontal_position(x_raw),
                    "y": get_highest_jump_y(y_raw),
                }
            )

    # ME burst-slider chains: the engine scores sc events per chain (the
    # head, already in _notes, plus sc-1 slices) but the dat file lists only
    # the head.  Emit the missing slices, evenly spaced head->tail in beat
    # AND position (measured from real replays: events land at exactly
    # b + k*(tb-b)/(sc-1)), or every note after the first chain pairs with
    # the wrong replay event.
    for c in (diff.get("_burstSliders") or diff.get("burstSliders") or []):
        hb = c.get("_time", c.get("b"))
        tb = c.get("_tailTime", c.get("tb"))
        sc = c.get("_sliceCount", c.get("sc", 1))
        try:
            sc = int(sc)
        except (TypeError, ValueError):
            sc = 1
        if hb is None or tb is None or sc < 2:
            continue
        hx = c.get("_lineIndex", c.get("x", 0))
        hy = c.get("_lineLayer", c.get("y", 0))
        tx = c.get("_tailLineIndex", c.get("tx", hx))
        ty = c.get("_tailLineLayer", c.get("ty", hy))
        col = c.get("_colorType", c.get("_type", c.get("c", 0)))
        cut = c.get("_cutDirection", c.get("d", 8))
        span = max(float(tb) - float(hb), 0.0)
        for k in range(1, sc):
            f = k / (sc - 1)
            map_notes.append(
                {
                    "kind": "note",
                    "beat": float(hb) + span * f,
                    "color": col,
                    "cut_dir": cut,
                    "x": get_horizontal_position(float(hx) + (float(tx) - float(hx)) * f),
                    "y": get_highest_jump_y(float(hy) + (float(ty) - float(hy)) * f),
                    "angle_offset": 0.0,
                }
            )

    # the positional pairing below sorts replay events by time; map notes
    # must be in the same order (dat files usually are, but slices just
    # appended are not)
    map_notes.sort(key=lambda n: float(n["beat"]))

    # v3 bombNotes are a separate array with no color field (implicit bombs)
    for n in bnotes:
        x_raw = n.get("_lineIndex", n.get("x", 0))
        y_raw = n.get("_lineLayer", n.get("y", 0))
        beat = n.get("_time", n.get("b", n.get("time", 0.0)))
        bombs.append(
            {
                "kind": "bomb",
                "beat": beat,
                "time": beat_to_sec(beat),
                "x": get_horizontal_position(x_raw),
                "y": get_highest_jump_y(y_raw),
            }
        )

    map_walls = []
    for w in walls:
        w_type = w.get("_type", w.get("type", 0))
        x_raw = w.get("_lineIndex", w.get("x", 0))
        dur = w.get("_duration", w.get("d", w.get("duration", 0.0)))
        beat = w.get("_time", w.get("b", w.get("time", 0.0)))
        width = w.get("_width", w.get("w", w.get("width", 1)))
        if w_type == 0:
            y, height = 0.0, 5.0
        elif w_type == 1:
            y, height = 2.0, 3.0
        else:
            continue
        map_walls.append(
            {
                "beat": beat,
                "time": beat_to_sec(beat),
                "end": beat_to_sec(beat + dur),
                "duration_beats": dur,
                "x": get_horizontal_position(x_raw),
                "y": get_highest_jump_y(y),
                "width": width,
                "height": height,
            }
        )

    return {
        "njs": njs,
        "jd": jd,
        "bpm": bpm,
        "notes": map_notes,
        "bombs": bombs,
        "walls": map_walls,
    }


def encode_note_into(note, array, count, time_now, event):
    """Encode one note following upstream Extensions.EncodeToArray (31 cols)."""
    idx = 0
    array[count, idx] = note["time"] - time_now
    idx += 1
    # 4 quality columns (sanitize: some replays carry Inf/NaN cut ratings)
    if event is not None:
        before = event["before_cut"] if np.isfinite(event["before_cut"]) else 0.0
        after = event["after_cut"] if np.isfinite(event["after_cut"]) else 0.0
        cut_dist = event["cut_dist"] if np.isfinite(event["cut_dist"]) else 1.0
        time_dev = event["time_dev"] if np.isfinite(event["time_dev"]) else 0.0
    else:
        before = after = cut_dist = time_dev = 0.0
    if event is not None and (before or after or cut_dist or time_dev):
        array[count, idx] = before / 2
        array[count, idx + 1] = after / 2
        array[count, idx + 2] = max(0.0, 1.0 - 3.0 * cut_dist)
        array[count, idx + 3] = time_dev
    elif event is not None:
        v = event["event_type"] / 5.0
        array[count, idx] = array[count, idx + 1] = array[count, idx + 2] = array[count, idx + 3] = v
    idx += 4

    if note["kind"] == "note":
        color = note["color"]
        x = np.clip(note["x"] / 2.0, -2.0, 2.0)
        y = np.clip(note["y"] / 2.0, -2.0, 2.0)
        rot = np.clip((CUTDIR_ANGLE.get(note["cut_dir"], 0.0) + note["angle_offset"]) / 360.0, -2.0, 2.0)
        if color == 0:
            array[count, idx] = x
            array[count, idx + 1] = y
            array[count, idx + 2] = rot
            idx += 3
            idx = one_hot(array, count, idx, note["cut_dir"], 0, 9)
            idx = one_hot(array, count, idx, -2, 0, 12)
        elif color == 1:
            idx = one_hot(array, count, idx, -2, 0, 12)
            array[count, idx] = x
            array[count, idx + 1] = y
            array[count, idx + 2] = rot
            idx += 3
            idx = one_hot(array, count, idx, note["cut_dir"], 0, 9)
    else:  # bomb
        array[count, idx] = np.clip(note["x"] / 2.0, -2.0, 2.0)
        array[count, idx + 1] = np.clip(note["y"] / 2.0, -2.0, 2.0)
        array[count, idx + 2] = 0
        idx += 3
        idx = one_hot(array, count, idx, 9, 0, 9)
        array[count, idx] = np.clip(note["x"] / 2.0, -2.0, 2.0)
        array[count, idx + 1] = np.clip(note["y"] / 2.0, -2.0, 2.0)
        array[count, idx + 2] = 0
        idx += 3
        idx = one_hot(array, count, idx, 9, 0, 9)


def one_hot(array, count, idx, value, min_v, limit):
    for i in range(min_v, limit + 1):
        array[count, idx] = 1.0 if i == value else 0.0
        idx += 1
    return idx


def encode_wall_into(wall, array, count, time_now):
    idx = 0
    array[count, idx] = max(wall["time"] - time_now, 0.0)
    idx += 1
    array[count, idx] = 1.0 if wall["time"] - time_now > 0 or wall["end"] - time_now < 0 else 0.0
    idx += 1
    array[count, idx] = wall["x"] / 2.0
    array[count, idx + 1] = wall["y"] / 2.0
    array[count, idx + 2] = wall["width"] / 2.0
    array[count, idx + 3] = wall["height"] / 2.0


def encode_hit_into(array, count, hit_time, time_now, cut):
    """Encode one note's hit target into y2 at slot `count`.

    cut: the replay note dict (with saber_type / cut_normal / result) or None
    if the note was never cut (e.g. missed). hit_time is seconds, absolute.
    Returns the clamped relative hit time (also used to decide window inclusion).
    """
    if cut is None:
        array[count, 0] = 1.0          # present: it is in the window but uncut
        array[count, 1] = -1.0         # saber unknown
        array[count, 2] = 0.0
        array[count, 3] = array[count, 4] = array[count, 5] = 0.0
        array[count, 6] = 0.0          # MISS
        return 0.0
    rel = hit_time - time_now
    array[count, 0] = 1.0
    array[count, 1] = float(cut.get("saber_type", -1))       # 0=left,1=right
    array[count, 2] = float(np.clip(rel, -1.0, 1.0))
    cn = cut.get("cut_normal", [0, 0, 0])
    array[count, 3] = float(np.clip(cn[0], -1.0, 1.0))
    array[count, 4] = float(np.clip(cn[1], -1.0, 1.0))
    array[count, 5] = float(np.clip(cn[2], -1.0, 1.0))
    array[count, 6] = float(cut.get("result", 0))
    return rel


def process_one(bsor_dir, map_dir, out_dir):
    """Extract windows for one parsed replay. Returns (name, status, n_windows) or None."""
    parsed = Path(bsor_dir)
    name = parsed.name
    try:
        frames = read_frames(parsed / "frames.f32")
        if len(frames) < FRAME_COUNT_INPUT + FRAME_COUNT_OUTPUT + 10:
            return name, "too-short", 0
        meta = load_meta(parsed)
        mapobj = load_map(meta, Path(map_dir))
        if mapobj is None:
            return name, "no-map", 0
        replay_notes = load_notes(parsed)

        times = frames[:, 0]
        # map note times in seconds: align to replay spawnTimes via map beat ordering
        # (upstream matches mapnote.Time == replaynote.spawnTime within 0.0005)
        map_notes = []
        for n in mapobj["notes"]:
            map_notes.append({**n, "time": None})
        # find replay spawn time for each map note by nearest beat-ordinal match
        spawn_times = sorted(rn["spawn_time"] for rn in replay_notes if rn["event_type"] != 3)
        if len(map_notes) != len(spawn_times):
            # A run that ended early (fail/quit) drops only the unspawned
            # TAIL: the prefix pairing stays correct.  Any other mismatch
            # interleaves with spawned times (map updated after the replay
            # was recorded, or a map element the parser does not model) and
            # shifts every note after it.  The engine spawns notes at exact
            # beat times, so pairing is VERIFIABLE: walk both lists in
            # lockstep and keep only the provably-aligned prefix.
            b2s = lambda b: float(b) * 60.0 / float(mapobj.get("bpm") or 120.0)
            tail_trunc = (
                len(map_notes) > len(spawn_times) and spawn_times
                and all(b2s(n["beat"]) > spawn_times[-1] + 0.25
                        for n in map_notes[len(spawn_times):])
            )
            if tail_trunc:
                print(f"  [trunc] {name}: run ended early; dropped "
                      f"{len(map_notes) - len(spawn_times)} unspawned tail notes",
                      flush=True)
                map_notes = map_notes[:len(spawn_times)]
            else:
                div = 0
                for mb, st in zip(map_notes, spawn_times):
                    if abs(b2s(mb["beat"]) - st) > 0.05:
                        break
                    div += 1
                print(f"  [desync] {name}: map_notes={len(map_notes)} "
                      f"replay_events={len(spawn_times)} aligned_prefix={div} "
                      f"hash={meta.get('hash', '')[:12]}", flush=True)
                if div < 20:
                    # nothing salvageable before the divergence
                    return name, "desync", 0
                map_notes = map_notes[:div]
                spawn_times = spawn_times[:div]
        for n, st in zip(map_notes, spawn_times):
            n["time"] = st
        map_notes = [n for n in map_notes if n["time"] is not None]

        # attach each note its OWN event.  Several notes can share a spawn
        # time (double notes, chain heads and slices), so a plain time->event
        # dict hands every one of them the FIRST event; consume one event
        # per note, in replay order, so a left/right double keeps its two
        # distinct sabers and cut normals.
        events_by_time = {}
        for rn in replay_notes:
            if rn["event_type"] != 3:
                events_by_time.setdefault(round(rn["spawn_time"], 4), []).append(rn)
        for n in map_notes:
            lst = events_by_time.get(round(n["time"], 4))
            n["event"] = lst.pop(0) if lst else None

        # all elements (notes + bombs) ordered by time, matching upstream allElements
        # bombs use beat-converted seconds (bomb encoding is purely positional)
        all_elements = []
        for n in map_notes:
            all_elements.append(
                {"kind": "note", "time": n["time"], "data": n}
            )
        for b in mapobj["bombs"]:
            if b.get("time") is not None:
                all_elements.append({"kind": "bomb", "time": b["time"], "data": b})
        all_elements = [e for e in all_elements if e["time"] is not None]
        all_elements.sort(key=lambda e: e["time"])

        # Build windows
        windows = []
        step = FRAME_COUNT_OUTPUT * 2
        i = FRAME_COUNT_INPUT + 10
        n_frames = len(frames)
        while i < n_frames - FRAME_COUNT_OUTPUT:
            time_now = times[i]
            x0 = np.zeros((FRAME_COUNT_INPUT, FRAME_LENGTH), dtype=np.float32)
            x1 = np.zeros(EXTRA_DETAILS_LENGTH, dtype=np.float32)
            x2 = np.zeros((NOTE_COUNT, NOTE_LENGTH), dtype=np.float32)
            x3 = np.zeros((WALL_COUNT, WALL_LENGTH), dtype=np.float32)
            y = np.zeros((FRAME_COUNT_OUTPUT, FRAME_OUT_LENGTH), dtype=np.float32)
            y2 = np.zeros((NOTE_COUNT, HIT_VEC), dtype=np.float32)

            # input frames (50) + output frames (50)
            # frames.f32 per frame: [t(f32), fps(i32), head(7), left(7), right(7)].
            # Transforms start at column 2; col 1 is the fps-as-float garbage that
            # must be skipped (it was previously included, shifting every channel
            # by one and dropping the right-hand quaternion's w component).
            for j in range(FRAME_COUNT_INPUT, -FRAME_COUNT_OUTPUT, -1):
                ci = i - j
                if j > 0:
                    # input: ToArray(arr, idx, curr) -> col0 = dt
                    f = frames[ci]
                    row = FRAME_COUNT_INPUT - j
                    x0[row, 0] = f[0] - time_now
                    x0[row, 1:] = f[2:23]
                else:
                    row = -j
                    x1[row] = frames[ci][0] - time_now
                    y[row] = frames[ci][2:23]

            # NaN/Inf compare False against any bound, so the range filter
            # below can't catch them.  Tracking-loss frames normalize their
            # quaternions to NaN (0/0) and would silently poison the batch
            # isfinite filters at train time; drop such windows here.
            if not (np.isfinite(x0).all() and np.isfinite(y).all()):
                i += step
                continue
            max_frame = max(np.abs(x0).max(), np.abs(y).max())
            if max_frame > 3.0:
                i += step
                continue

            # extra stats
            mods = (meta.get("modifiers") or "").lower()
            time_scale = 1.0
            if "ss" in mods:
                time_scale = 0.85
            elif "fs" in mods:
                time_scale = 1.2
            elif "sf" in mods:
                time_scale = 1.5
            if any(x in mods for x in ("na", "nf", "no", "nb", "sa")):
                i += step
                continue
            x1[FRAME_COUNT_OUTPUT + 0] = mapobj["njs"] / 30 * time_scale
            x1[FRAME_COUNT_OUTPUT + 1] = meta.get("jumpDistance", 0.0) / 30
            x1[FRAME_COUNT_OUTPUT + 2] = meta.get("height", 0.0)
            x1[FRAME_COUNT_OUTPUT + 3] = 1.0 if "pm" in mods else 0.0
            x1[FRAME_COUNT_OUTPUT + 4] = 1.0 if "sc" in mods else 0.0
            x1[FRAME_COUNT_OUTPUT + 5] = 1.0 if "od" in mods else 0.0

            # upcoming notes (within 1.5s) -- iterate all_elements with mapNoteIter
            count = 0
            map_note_iter = 0
            while (map_note_iter < len(all_elements)
                   and time_now - 0.0 > all_elements[map_note_iter]["time"]):
                map_note_iter += 1
            for j in range(map_note_iter, len(all_elements)):
                dt = all_elements[j]["time"] - time_now
                if MAX_TIME_AHEAD >= dt >= 0:
                    ev = None
                    if all_elements[j]["kind"] == "note":
                        # this note's own event (attached above; distinct for
                        # same-beat doubles), falling back to any same-beat
                        # event for notes whose event was consumed elsewhere
                        ev = all_elements[j]["data"].get("event")
                        if ev is None:
                            lst = events_by_time.get(round(all_elements[j]["time"], 4))
                            ev = lst[0] if lst else None
                    encode_note_into(all_elements[j]["data"], x2, count, time_now, ev)
                    count += 1
                    if count == NOTE_COUNT:
                        break
                elif dt > MAX_TIME_AHEAD:
                    break
            if count == 0:
                i += step
                continue

            # hit-target y2: notes whose CUT event lands within the output window.
            # out_end is the last output frame's time.
            hi = min(i + FRAME_COUNT_OUTPUT, n_frames - 1)
            out_end = frames[hi][0]
            hit_count = 0
            for j in range(map_note_iter, len(all_elements)):
                if all_elements[j]["kind"] != "note":
                    continue
                spawn = all_elements[j]["time"]
                if spawn - time_now > MAX_TIME_AHEAD:
                    break
                ev = all_elements[j]["data"].get("event")
                if ev is None:
                    lst = events_by_time.get(round(spawn, 4))
                    ev = lst[0] if lst else None
                if ev is not None:
                    hit_time = ev["event_time"]
                    if time_now <= hit_time <= out_end:
                        if hit_count == NOTE_COUNT:
                            break
                        encode_hit_into(y2, hit_count, hit_time, time_now, ev)
                        hit_count += 1
                else:
                    # uncut note that spawns inside the window -> MISS present
                    if time_now <= spawn <= out_end and hit_count < NOTE_COUNT:
                        encode_hit_into(y2, hit_count, spawn, time_now, None)
                        hit_count += 1

            # upcoming walls (within 1.5s)
            wall_count = 0
            for w in mapobj["walls"]:
                if w["time"] - time_now <= MAX_TIME_AHEAD and w["end"] - time_now >= 0:
                    encode_wall_into(w, x3, wall_count, time_now)
                    wall_count += 1
                    if wall_count == WALL_COUNT:
                        break
                elif w["time"] - time_now > MAX_TIME_AHEAD:
                    break

            windows.append((x0, x1, x2, x3, y, y2))
            i += step

        if not windows:
            return name, "no-windows", 0

        # Pack
        n = len(windows)
        x0a = np.stack([w[0] for w in windows])
        x1a = np.stack([w[1] for w in windows])
        x2a = np.stack([w[2] for w in windows])
        x3a = np.stack([w[3] for w in windows])
        ya = np.stack([w[4] for w in windows])
        y2a = np.stack([w[5] for w in windows])
        # power-loss-safe publish: a partial npz must never exist under the
        # final name (the processor resume path trusts its existence).
        # tmp ends in .npz so numpy doesn't append a second extension.
        tmp = out_dir / f".{name}.tmp.npz"
        np.savez_compressed(
            tmp,
            x0=x0a, x1=x1a, x2=x2a, x3=x3a, y=ya, y2=y2a,
        )
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, out_dir / f"{name}.npz")
        return name, "ok", n
    except Exception as e:
        return name, f"err:{type(e).__name__}:{e}", 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["data_dir"])
    parsed_dir = root / cfg["parsed_dir"]
    window_dir = root / cfg["window_dir"]
    map_dir = root / cfg["map_dir"]
    window_dir.mkdir(parents=True, exist_ok=True)

    dirs = sorted(
        d for d in parsed_dir.iterdir()
        if d.is_dir() and (d / "frames.f32").exists()
    )
    if args.limit:
        dirs = dirs[: args.limit]
    if not dirs:
        print("no parsed replays found")
        return

    print(f"windowing {len(dirs)} replays with {args.workers} workers...", flush=True)
    stats = {}
    total_windows = 0
    done = 0
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, d, map_dir, window_dir): d for d in dirs
        }
        for fut in as_completed(futures):
            name, status, nw = fut.result()
            stats[status] = stats.get(status, 0) + 1
            total_windows += nw
            done += 1
            if done % 200 == 0 or done == len(dirs):
                elapsed = time.time() - start
                print(
                    f"  {done}/{len(dirs)} ({elapsed:.0f}s) "
                    f"ok={stats.get('ok', 0)} bad={sum(v for k, v in stats.items() if k != 'ok')} "
                    f"windows={total_windows}",
                    flush=True,
                )
    print("window done:", stats, "total_windows:", total_windows)


if __name__ == "__main__":
    main()
