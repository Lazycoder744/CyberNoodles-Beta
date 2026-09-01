"""Data pipelines for the Beat Saber training system.

- WindowShardDataset: streaming IterableDataset over /mnt/games/bs-ai/windows/*.npz
  (port of upstream WallsIterableDataset: in-place x2 smoothing + filter mask).
- ScoreMaxSong: whole-song roll-out harness (stride-50 tiling) that builds a
  fully-differentiable generated trajectory and scores ALL map notes with
  torchsaber.SaberScorer -> -smooth_total / note_count as the score-max loss.
  Supports both replay-backed songs and map-only evals (no replay file).
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset
from types import SimpleNamespace

from window import (
    encode_note_into, encode_wall_into,
    get_horizontal_position, get_highest_jump_y,
)
# Optional training-side dependencies: only the score-max optimization /
# flow-smoothing paths use them. The ONNX generation path in this release
# (ScoreMaxSong loading + encode_window) does not; imports fail loudly ONLY
# if one of those methods is actually called.
try:
    import bsgen
except ImportError:
    bsgen = None
try:
    from torchsaber import SaberScorer, q_rot
except ImportError:
    SaberScorer = q_rot = None

NOTE_COUNT = 50
NOTE_LENGTH = 31
WALL_COUNT = 50
WALL_LENGTH = 6
FRAME_COUNT_INPUT = 50
FRAME_COUNT_OUTPUT = 50
FRAME_LENGTH = 22
EXTRA_STATS = 6
NUM_TARGET_CHANNELS = 21
MAX_TIME_AHEAD = 1.5
EPS = 1e-6


# -------------------------
# Mini map parser (replaces interpretMapFiles.create_map, which lived in /tmp)
# -------------------------

def _g(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def create_map(map_dir):
    """Parse a Beat Saber map directory into an interpretMapFiles-like object.

    Handles v2 (underscored) and v3 (short) Info.dat / difficulty formats.
    Returns SimpleNamespace(beatsPerMinute, beatMaps{mode:{diff:Beatmap}}, song_meta).
    Each Beatmap: SimpleNamespace(noteJumpMovementSpeed, notes[], obstacles[]).
    """
    map_dir = os.fspath(map_dir)
    info_path = None
    for cand in ("Info.dat", "info.dat", "INFO.dat"):
        p = os.path.join(map_dir, cand)
        if os.path.exists(p):
            info_path = p
            break
    if info_path is None:
        raise FileNotFoundError("Info.dat not found in %s" % map_dir)
    with open(info_path) as f:
        info = json.load(f)

    bpm = float(_g(info, "_beatsPerMinute", "beatsPerMinute") or 120.0)
    song_meta = {
        "songName": _g(info, "_songName", "songName") or "",
        "mapper": _g(info, "_levelAuthorName", "levelAuthorName") or "",
        "environment": _g(info, "_environmentName", "environmentName") or "DefaultEnvironment",
    }
    sets = _g(info, "_difficultyBeatmapSets", "difficultyBeatmapSets") or []
    beatmaps = {}
    for s in sets:
        mode = _g(s, "_beatmapCharacteristicName", "beatmapCharacteristicName") or "Standard"
        for b in (_g(s, "_difficultyBeatmaps", "difficultyBeatmaps") or []):
            diff = _g(b, "_difficulty", "difficulty")
            fname = _g(b, "_beatmapFilename", "beatmapFilename")
            njs = float(_g(b, "_noteJumpMovementSpeed", "noteJumpMovementSpeed") or 10.0)
            dat_path = os.path.join(map_dir, fname)
            if not os.path.exists(dat_path):
                continue
            with open(dat_path) as f:
                dat = json.load(f)

            notes = []
            # v2: _notes (type 0/1 note, 3 bomb) ; v3: colorNotes + bombNotes
            for n in (_g(dat, "_notes", "notes") or []):
                nt = _g(n, "_type", "c", "type")
                if nt not in (0, 1, 3):
                    continue
                cd = _g(n, "_customData") or {}
                # NE fake notes are decoration: never spawned, never scored
                if cd.get("_fake") or cd.get("fake"):
                    continue
                # NE float _position: [x, y] in lineIndex/lineLayer space
                pos = cd.get("_position")
                fx = float(pos[0]) if pos else None
                fy = float(pos[1]) if pos else None
                notes.append(SimpleNamespace(
                    type=nt,
                    time=float(_g(n, "_time", "b", "time") or 0.0),
                    cutDirection=int(_g(n, "_cutDirection", "d", "cutDirection") or 8),
                    lineIndex=int(round(float(_g(n, "_lineIndex", "x", "lineIndex") or 0))),
                    lineLayer=int(round(float(_g(n, "_lineLayer", "y", "lineLayer") or 0))),
                    fx=fx, fy=fy,
                ))
            for n in (_g(dat, "colorNotes") or []):
                notes.append(SimpleNamespace(
                    type=int(_g(n, "c", "type") or 0),
                    time=float(_g(n, "b", "time") or 0.0),
                    cutDirection=int(_g(n, "d", "cutDirection") or 8),
                    lineIndex=int(_g(n, "x", "lineIndex") or 0),
                    lineLayer=int(_g(n, "y", "lineLayer") or 0),
                    fx=None, fy=None,
                ))
            for n in (_g(dat, "bombNotes") or []):
                notes.append(SimpleNamespace(
                    type=3,
                    time=float(_g(n, "b", "time") or 0.0),
                    cutDirection=8,
                    lineIndex=int(_g(n, "x", "lineIndex") or 0),
                    lineLayer=int(_g(n, "y", "lineLayer") or 0),
                    fx=None, fy=None,
                ))
            # ME burst-slider chains: engine scores sc events per chain
            # (head + sc-1 slices), the dat lists only the head.  Emit the
            # missing slices evenly spaced head->tail so generation sees
            # (and aims at) every scored element.
            for c in (_g(dat, "_burstSliders", "burstSliders") or []):
                hb = _g(c, "_time", "b")
                tb = _g(c, "_tailTime", "tb")
                sc = _g(c, "_sliceCount", "sc")
                try:
                    sc = int(sc)
                except (TypeError, ValueError):
                    sc = 1
                if hb is None or tb is None or sc < 2:
                    continue
                hx = float(_g(c, "_lineIndex", "x") or 0)
                hy = float(_g(c, "_lineLayer", "y") or 0)
                tx = float(_g(c, "_tailLineIndex", "tx") or hx)
                ty = float(_g(c, "_tailLineLayer", "ty") or hy)
                col = int(_g(c, "_colorType", "_type", "c") or 0)
                cut = int(_g(c, "_cutDirection", "d") or 8)
                span = max(float(tb) - float(hb), 0.0)
                for k in range(1, sc):
                    f = k / (sc - 1)
                    notes.append(SimpleNamespace(
                        type=col,
                        time=float(hb) + span * f,
                        cutDirection=cut,
                        lineIndex=int(round(hx + (tx - hx) * f)),
                        lineLayer=int(round(hy + (ty - hy) * f)),
                        fx=hx + (tx - hx) * f, fy=hy + (ty - hy) * f,
                    ))
            notes.sort(key=lambda n: n.time)

            obstacles = []
            # v2: _obstacles (type 0 full, 1 crouch) ; v3: obstacles (x,y,d,w,h)
            for w in (_g(dat, "_obstacles", "obstacles") or []):
                w_type = _g(w, "_type", "c", "type")
                if w_type is None:
                    h = float(_g(w, "_height", "h", "height") or 5.0)
                    w_type = 0 if h >= 4.0 else 1
                obstacles.append(SimpleNamespace(
                    type=int(w_type),
                    time=float(_g(w, "_time", "b", "time") or 0.0),
                    duration=float(_g(w, "_duration", "d", "duration") or 0.0),
                    lineIndex=int(_g(w, "_lineIndex", "x", "lineIndex") or 0),
                    width=int(_g(w, "_width", "w", "width") or 1),
                    height=int(_g(w, "_height", "h", "height") or (5 if int(w_type) == 0 else 3)),
                ))
            beatmaps.setdefault(mode, {})[diff] = SimpleNamespace(
                noteJumpMovementSpeed=njs, notes=notes, obstacles=obstacles)
    return SimpleNamespace(beatsPerMinute=bpm, beatMaps=beatmaps, song_meta=song_meta)


# -------------------------
# Imitation data
# -------------------------

def _filter_mask_batch(frames_in, y_t):
    dx = np.max(np.abs(frames_in[:, :, 1] - y_t[:, :, 0]), axis=1)
    dy = np.max(np.abs(frames_in[:, :, 2] - y_t[:, :, 1]), axis=1)
    dz = np.max(np.abs(frames_in[:, :, 3] - y_t[:, :, 2]), axis=1)
    mask = (dx < 0.25) & (dy < 0.25) & (dz < 0.25)
    mask &= np.isfinite(frames_in).all(axis=(1, 2))
    mask &= np.isfinite(y_t).all(axis=(1, 2))
    return mask


def list_window_shards(window_dir, cap=0):
    shards = sorted(glob.glob(os.path.join(window_dir, "*.npz")))
    if cap and len(shards) > cap:
        shards = shards[:cap]
    return shards


class WindowShardDataset(IterableDataset):
    """Streams (x0, frame_times, extra_stats, x2, x3, tdiff), y batches from .npz shards."""

    def __init__(self, shards, batch_size, repeat=True, shuffle_files=True, shuffle_within=True, seed=72583):
        super().__init__()
        self.shards = list(shards)
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle_files = shuffle_files
        self.shuffle_within = shuffle_within
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = np.random.default_rng(self.seed + wid)
        while True:
            order = np.arange(len(self.shards))
            if self.shuffle_files:
                rng.shuffle(order)
            for idx in order:
                z = np.load(self.shards[idx])
                x0, x1, x2, x3, y = z["x0"], z["x1"], z["x2"], z["x3"], z["y"]
                timings = x2[:, :, 4].astype(np.float32, copy=True)

                weights = np.maximum(
                    0, 1 - 25 * np.abs(x2[:, :, [0]] - np.reshape(x2[:, :, [0]], (len(x2), 1, NOTE_COUNT)))
                ) * (np.where(x2[:, :, [0]] == 0, [0], [1])
                     * np.reshape(np.where(x2[:, :, [4]] == 0, [0], [1]), (len(x2), 1, NOTE_COUNT)))
                denom = np.sum(weights, axis=2) + EPS
                x2[:, :, 1] = np.sum(np.reshape(x2[:, :, [1]], (len(x2), 1, NOTE_COUNT)) * weights, axis=2) / denom
                x2[:, :, 2] = np.sum(np.reshape(x2[:, :, [2]], (len(x2), 1, NOTE_COUNT)) * weights, axis=2) / denom
                x2[:, :, 4] = np.sum(np.reshape(x2[:, :, [4]], (len(x2), 1, NOTE_COUNT)) * weights, axis=2) / denom

                B = x0.shape[0]
                indices = np.arange(B)
                if self.shuffle_within:
                    rng.shuffle(indices)

                for s in range(0, B, self.batch_size):
                    sl = indices[s:s + self.batch_size]
                    frames_in = x0[sl, :FRAME_COUNT_INPUT, :FRAME_LENGTH].astype(np.float32, copy=False)
                    frame_times = x1[sl, :FRAME_COUNT_OUTPUT][..., np.newaxis].astype(np.float32, copy=False)
                    extra_stats = x1[sl, FRAME_COUNT_OUTPUT:].astype(np.float32, copy=False)
                    notes_in = x2[sl, :NOTE_COUNT, :NOTE_LENGTH].astype(np.float32, copy=False)
                    walls_in = x3[sl, :WALL_COUNT, :WALL_LENGTH].astype(np.float32, copy=False)
                    note_time_diffs = timings[sl, :].astype(np.float32, copy=False)
                    y_t = y[sl, :FRAME_COUNT_OUTPUT, :NUM_TARGET_CHANNELS].astype(np.float32, copy=False)

                    keep = _filter_mask_batch(frames_in, y_t)
                    if not np.any(keep):
                        continue

                    inputs = (
                        torch.from_numpy(frames_in[keep]),
                        torch.from_numpy(frame_times[keep]),
                        torch.from_numpy(extra_stats[keep]),
                        torch.from_numpy(notes_in[keep]),
                        torch.from_numpy(walls_in[keep]),
                        torch.from_numpy(note_time_diffs[keep]),
                    )
                    yield inputs, torch.from_numpy(y_t[keep])

            if not self.repeat:
                break


# -------------------------
# Score-max: whole-song roll-out
# -------------------------

class ScoreMaxSong:
    """Stitch a whole song from model predictions and score ALL map notes.

    Anchors at i = seed + k*FRAME_COUNT_OUTPUT. Window k's 50 input frames are the
    PAST 50 frames (reference seed for the first window, otherwise the previous
    window's predictions). x2 note times use hit-time convention (= beat-converted
    note time), matching the imitation training distribution.
    """

    def __init__(self, map_dir, replay_path, device, dtype=torch.float64, max_windows=0,
                 mode=None, difficulty=None, song_hash=None, fps=90.0):
        """ScoreMaxSong for a map directory.

        replay_path: optional .bsor replay. If None, runs in map-only mode:
        requires mode + difficulty; uses a synthetic 90 FPS frame grid, a neutral
        reference pose, JD=NJS (1s reaction) and height=1.8.
        """
        from bsor.Bsor import make_bsor

        self.device = device
        self.dtype = dtype
        self.max_windows = max_windows
        self.map_dir = os.fspath(map_dir)
        self.replay_path = os.fspath(replay_path) if replay_path else None
        self.replay_info = None

        m = create_map(map_dir)

        if replay_path is None:
            if not mode or not difficulty:
                raise ValueError("map-only mode requires mode and difficulty")
            info_mode, info_diff = mode, difficulty
            r = None
        else:
            with open(replay_path, "rb") as f:
                r = make_bsor(f)
            info_mode, info_diff = r.info.mode, r.info.difficulty

        if info_mode not in m.beatMaps or info_diff not in m.beatMaps[info_mode]:
            raise KeyError("no %s/%s beatmap in %s" % (info_mode, info_diff, map_dir))
        bm = m.beatMaps[info_mode][info_diff]
        self.mode = info_mode
        self.difficulty = info_diff

        self.bpm = float(m.beatsPerMinute)
        self.NJS = float(bm.noteJumpMovementSpeed)

        if r is None:
            self.JD = self.NJS            # 1s reaction window
            self.height = 1.8
            mods = ""
            self.song_hash = song_hash or ""
            self.song_name = m.song_meta.get("songName", "")
            self.mapper = m.song_meta.get("mapper", "")
            self.environment = m.song_meta.get("environment", "")
        else:
            self.JD = float(r.info.jumpDistance)
            self.height = float(r.info.height)
            mods = (r.info.modifiers or "").lower()
            self.replay_info = r.info
            self.song_hash = r.info.songHash or song_hash or ""
            self.song_name = r.info.songName or ""
            self.mapper = r.info.mapper or ""
            self.environment = r.info.environment or ""

        time_scale = 0.85 if "ss" in mods else (1.2 if "fs" in mods else (1.5 if "sf" in mods else 1.0))
        self.stats = np.array([
            self.NJS / 30 * time_scale, self.JD / 30, self.height,
            1.0 if "pm" in mods else 0.0,
            1.0 if "sc" in mods else 0.0,
            1.0 if "od" in mods else 0.0,
        ], dtype=np.float32)

        if r is None:
            # map-only: synthetic 90 FPS grid covering the map duration
            beat_to_sec = lambda b: b * 60.0 / self.bpm
            note_times = sorted(beat_to_sec(n.time) for n in bm.notes)
            last = note_times[-1] if note_times else 10.0
            t_end = last + self.JD / self.NJS + 1.0
            t = np.arange(0.0, t_end, 1.0 / fps)
            F = len(t)
            neutral = np.zeros((F, FRAME_LENGTH), dtype=np.float64)
            neutral[:, 0] = t
            # neutral pose: head at origin-ish facing +z, sabers at rest
            neutral[:, 1:4] = [0.0, 1.7, 0.0]
            neutral[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
            neutral[:, 8:11] = [-0.4, 1.2, 0.4]
            neutral[:, 10:14] = [0.0, 0.0, 0.0, 1.0]
            neutral[:, 15:18] = [0.4, 1.2, 0.4]
            neutral[:, 17:21] = [0.0, 0.0, 0.0, 1.0]
            self.fps = float(fps)
            self.SF_ref = neutral
        else:
            frames = r.frames[1:]
            F = len(frames)
            t = np.array([fr.time for fr in frames], dtype=np.float64)
            self.fps = float(np.median([fr.fps for fr in r.frames if fr.fps > 0])) if r.frames else float(fps)

            # full frame transforms in window.py layout (col0 = time, cols 1-21 = head7/left7/right7)
            SF = np.zeros((F, FRAME_LENGTH), dtype=np.float64)
            SF[:, 0] = t
            for i, fr in enumerate(frames):
                SF[i, 1:8] = [fr.head.x, fr.head.y, fr.head.z,
                              fr.head.x_rot, fr.head.y_rot, fr.head.z_rot, fr.head.w_rot]
                SF[i, 8:15] = [fr.left_hand.x, fr.left_hand.y, fr.left_hand.z,
                               fr.left_hand.x_rot, fr.left_hand.y_rot, fr.left_hand.z_rot, fr.left_hand.w_rot]
                SF[i, 15:22] = [fr.right_hand.x, fr.right_hand.y, fr.right_hand.z,
                                fr.right_hand.x_rot, fr.right_hand.y_rot, fr.right_hand.z_rot, fr.right_hand.w_rot]
            self.SF_ref = SF
        self.F = F
        self.t = t

        # all map notes (type 0/1) -> scorer note arrays (hit-time convention)
        beat_to_sec = lambda b: b * 60.0 / self.bpm
        self.notes = []
        self.bombs = []
        for n in bm.notes:
            if n.type == 0 or n.type == 1:
                self.notes.append({
                    "time": beat_to_sec(n.time),
                    "kind": "note", "color": n.type, "cut_dir": n.cutDirection,
                    "line_index": n.lineIndex, "line_layer": n.lineLayer,
                    # NE float positions (and interpolated chain slices) live
                    # in the same grid space; use them when present so
                    # scoring/aim matches the engine's actual spawn point
                    "x": get_horizontal_position(n.fx if n.fx is not None else n.lineIndex),
                    "y": get_highest_jump_y(n.fy if n.fy is not None else n.lineLayer),
                    "angle_offset": 0.0,
                })
            elif n.type == 3:
                self.bombs.append({
                    "kind": "bomb", "time": beat_to_sec(n.time),
                    "x": get_horizontal_position(n.fx if n.fx is not None else n.lineIndex),
                    "y": get_highest_jump_y(n.fy if n.fy is not None else n.lineLayer),
                })
        self.notes.sort(key=lambda n: n["time"])
        self.bombs.sort(key=lambda b: b["time"])
        self.walls = []
        for w in bm.obstacles:
            if w.type == 0:
                y, height = 0.0, 5.0
            elif w.type == 1:
                y, height = 2.0, 3.0
            else:
                continue
            self.walls.append({
                "time": beat_to_sec(w.time),
                "end": beat_to_sec(w.time + w.duration),
                "x": get_horizontal_position(w.lineIndex),
                "y": get_highest_jump_y(y),
                "width": float(w.width), "height": float(height),
            })
        self.walls.sort(key=lambda w: w["time"])

        self.note_time = np.array([n["time"] for n in self.notes], dtype=np.float64)
        self.line_index = np.array([n["line_index"] for n in self.notes], dtype=np.int64)
        self.line_layer = np.array([n["line_layer"] for n in self.notes], dtype=np.int64)
        self.ntype = np.array([n["color"] for n in self.notes], dtype=np.int64)
        self.cut_dir = np.array([n["cut_dir"] for n in self.notes], dtype=np.int64)
        self.r_e = np.clip(np.searchsorted(t, self.note_time, side="right"), 0, F - 1)

        # swing window length J (same as torchsaber.preprocess)
        J = 2
        for f in range(F):
            c = 0
            j = f - 1
            while j >= 0 and (t[f] - t[j + 1] < 0.4):
                c += 1
                j -= 1
            J = max(J, c + 2)
        self.J = min(J, 501)

    def __len__(self):
        return self.F

    # -- window encoding -----------------------------------------------------
    def _encode_window(self, i, ctx21):
        """Build (x0, frame_times, extra_stats, x2, x3) float32 tensors for anchor i.

        ctx21: (50, 21) tensor of past-frame transforms (no grad).
        """
        return self._encode_window_batch(i, ctx21.unsqueeze(0))

    def _encode_window_batch(self, i, ctx21K):
        """Batched _encode_window. ctx21K: (K, 50, 21) past-frame transforms.

        The map context (notes/walls/stats/out-times) is shared across the batch;
        only the 50-frame history x0 differs per candidate. Returns tensors with
        batch dim K.
        """
        t = self.t
        time_now = t[i]

        K = ctx21K.shape[0]
        x0 = torch.zeros((K, FRAME_COUNT_INPUT, FRAME_LENGTH), dtype=torch.float32, device=self.device)
        x0[:, :, 0] = torch.from_numpy(t[i - FRAME_COUNT_INPUT:i] - time_now).to(self.device).to(torch.float32)
        x0[:, :, 1:] = ctx21K.to(torch.float32)

        out_times = t[i:i + FRAME_COUNT_OUTPUT] - time_now

        x2 = np.zeros((NOTE_COUNT, NOTE_LENGTH), dtype=np.float32)
        count = 0
        for n in self.notes:
            dt = n["time"] - time_now
            if MAX_TIME_AHEAD >= dt >= 0:
                encode_note_into(n, x2, count, time_now, None)
                count += 1
                if count == NOTE_COUNT:
                    break
            elif dt > MAX_TIME_AHEAD:
                break
        for b in self.bombs:
            dt = b["time"] - time_now
            if MAX_TIME_AHEAD >= dt >= 0:
                encode_note_into(b, x2, count, time_now, None)
                count += 1
                if count == NOTE_COUNT:
                    break
            elif dt > MAX_TIME_AHEAD:
                break

        x3 = np.zeros((WALL_COUNT, WALL_LENGTH), dtype=np.float32)
        wcount = 0
        for w in self.walls:
            if w["time"] - time_now <= MAX_TIME_AHEAD and w["end"] - time_now >= 0:
                encode_wall_into(w, x3, wcount, time_now)
                wcount += 1
                if wcount == WALL_COUNT:
                    break
            elif w["time"] - time_now > MAX_TIME_AHEAD:
                break

        return (
            x0,
            torch.from_numpy(out_times.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(self.device).expand(K, -1, -1),
            torch.from_numpy(self.stats).unsqueeze(0).to(self.device).expand(K, -1),
            torch.from_numpy(x2).unsqueeze(0).to(self.device).expand(K, -1, -1),
            torch.from_numpy(x3).unsqueeze(0).to(self.device).expand(K, -1, -1),
        )

    def roll_out(self, model, train=False):
        """Autoregressive stitch. Returns (smooth_total, G, scorer_out)."""
        if train:
            model.train()
        else:
            model.eval()
        F = self.F
        first = FRAME_COUNT_INPUT
        n_windows = max(0, (F - 1 - first) // FRAME_COUNT_OUTPUT)
        if self.max_windows:
            n_windows = min(n_windows, self.max_windows)

        seed21 = torch.tensor(self.SF_ref[:FRAME_COUNT_INPUT, 1:22], device=self.device, dtype=torch.float32)

        preds = []
        for k in range(n_windows):
            i = first + k * FRAME_COUNT_OUTPUT
            ctx21 = seed21 if k == 0 else preds[k - 1].detach()
            x0, ft, stats, x2, x3 = self._encode_window(i, ctx21)
            pred = model(x0, ft, stats, x2, x3, torch.zeros_like(x2[:, :, 4]))
            preds.append(pred.squeeze(0).to(self.dtype))

        end_i = first + n_windows * FRAME_COUNT_OUTPUT
        parts = [torch.tensor(self.SF_ref[:FRAME_COUNT_INPUT, 1:22], device=self.device, dtype=self.dtype)]
        parts += preds
        if end_i < F:
            parts.append(torch.tensor(self.SF_ref[end_i:, 1:22], device=self.device, dtype=self.dtype))
        G_transforms = torch.cat(parts, dim=0)
        t_col = torch.from_numpy(self.t).to(device=self.device, dtype=self.dtype).unsqueeze(1)
        G = torch.cat([t_col, G_transforms], dim=1)

        out = self._score(G)
        return out.smooth_total, G, out

    def _score(self, G):
        qL = G[:, 11:15]
        qR = G[:, 18:22]
        nL = torch.linalg.vector_norm(qL, dim=-1, keepdim=True).clamp_min(1e-12)
        nR = torch.linalg.vector_norm(qR, dim=-1, keepdim=True).clamp_min(1e-12)
        qL = qL / nL
        qR = qR / nR
        z = torch.zeros_like(G[:, 1:4])
        z[:, 2] = 1.0
        hL = G[:, 8:11]
        hR = G[:, 15:18]
        tipL = hL + q_rot(z, qL)
        tipR = hR + q_rot(z, qR)

        N = len(self.note_time)
        D = {
            "t": torch.from_numpy(self.t).to(device=self.device, dtype=self.dtype),
            "head": G[:, 1:4],
            "hiltL": hL, "tipL": tipL, "hiltR": hR, "tipR": tipR,
            "note_time": torch.from_numpy(self.note_time).to(device=self.device, dtype=self.dtype),
            "line_index": torch.from_numpy(self.line_index).to(device=self.device),
            "line_layer": torch.from_numpy(self.line_layer).to(device=self.device),
            "ntype": torch.from_numpy(self.ntype).to(device=self.device),
            "cut_dir": torch.from_numpy(self.cut_dir).to(device=self.device),
            "e_idx": torch.arange(N, device=self.device),
            "n_idx": torch.arange(N, device=self.device),
            "r_e": torch.from_numpy(self.r_e).to(device=self.device),
            "hand": torch.from_numpy(self.ntype).to(device=self.device),
            "cut_point": torch.zeros((N, 3), device=self.device, dtype=self.dtype),
            "J": self.J, "NJS": self.NJS, "bpm": self.bpm, "JD": self.JD, "height": self.height,
        }
        scorer = SaberScorer(device=self.device, dtype=self.dtype).to(self.device)
        out = scorer(D)
        return out

    def reference_score(self):
        """Score the reference trajectory through the same all-notes path."""
        with torch.no_grad():
            G = torch.tensor(self.SF_ref, device=self.device, dtype=self.dtype)
        return self._score(G).smooth_total

    def roll_out_flow(self, model, n_steps=16, K=4, temp=1.0, train=False, t_end=1.0):
        """Flow-selection rollout: sample K whole-song trajectories from the flow policy.

        Each candidate integrates the ODE from fresh noise per window (stochastic ->
        different replays). Scores each candidate with the differentiable scorer and
        returns (softmax_weighted_loss, best_G, scores, G_list). When train=True the
        score graph is kept for backprop; loss = -sum_k w_k * smooth_k / n_notes.
        """
        import torch.utils.checkpoint as ckpt

        if train:
            model.train()
        else:
            model.eval()
        F = self.F
        first = FRAME_COUNT_INPUT
        n_windows = max(0, (F - 1 - first) // FRAME_COUNT_OUTPUT)
        if self.max_windows:
            n_windows = min(n_windows, self.max_windows)

        def flow_window(x0, ft, stats, x2, x3, noise):
            ctx = model.encode_context(x0, ft, stats, x2, x3)
            return model.sample(ctx, noise=noise, n_steps=n_steps, t_end=t_end)

        seed21 = torch.tensor(self.SF_ref[:FRAME_COUNT_INPUT, 1:22], device=self.device, dtype=torch.float32)

        sigma = bsgen.FLOW_SIGMA.to(self.device)

        Gs = []
        smooths = []
        # Batched rollout: all K candidates share one map context per window and
        # only differ in their 50-frame history (x0) and noise, so we encode and
        # ODE-sample them together as a K-batch instead of K separate rollouts.
        preds = []  # list of (K, 50, 21), one entry per window
        for k in range(n_windows):
            i = first + k * FRAME_COUNT_OUTPUT
            if k == 0:
                ctx21K = seed21.unsqueeze(0).expand(K, -1, -1)
            else:
                ctx21K = preds[k - 1].detach()
            x0, ft, stats, x2, x3 = self._encode_window_batch(i, ctx21K)
            noise = torch.randn(K, FRAME_COUNT_OUTPUT, NUM_TARGET_CHANNELS,
                                device=self.device, dtype=torch.float32) * sigma
            if train:
                pred = ckpt.checkpoint(flow_window, x0, ft, stats, x2, x3, noise,
                                       use_reentrant=False)
            else:
                pred = flow_window(x0, ft, stats, x2, x3, noise)
            preds.append(pred.to(self.dtype))

        end_i = first + n_windows * FRAME_COUNT_OUTPUT
        head = torch.tensor(self.SF_ref[:FRAME_COUNT_INPUT, 1:22], device=self.device, dtype=self.dtype)
        tail = None
        if end_i < F:
            tail = torch.tensor(self.SF_ref[end_i:, 1:22], device=self.device, dtype=self.dtype)
        for c in range(K):
            parts = [head] + [p[c] for p in preds]
            if tail is not None:
                parts.append(tail)
            G_transforms = torch.cat(parts, dim=0)
            t_col = torch.from_numpy(self.t).to(device=self.device, dtype=self.dtype).unsqueeze(1)
            G = torch.cat([t_col, G_transforms], dim=1)
            out = self._score(G)
            Gs.append(G)
            smooths.append(out.smooth_total)

        scores = torch.stack(smooths) if K > 1 else smooths[0]
        if K > 1:
            w = torch.softmax(scores / max(temp, 1e-6), dim=0)
            loss = -(w * scores).sum() / max(1, len(self.notes))
            best_idx = int(scores.argmax().item())
        else:
            loss = -scores / max(1, len(self.notes))
            best_idx = 0
        return loss, Gs[best_idx], scores.detach(), Gs

    # -- eval artifacts ------------------------------------------------------
    def write_replay_bsor(self, path, G):
        """Write a generated trajectory G (F, 22) as a playable .bsor file."""
        from bsor.Bsor import Info, VRObject, Frame
        from bsor.Bsor import Bsor

        ri = self.replay_info
        info = Info()
        info.version = getattr(ri, "version", "") or ""
        info.gameVersion = getattr(ri, "gameVersion", "1.34.0")
        info.timestamp = getattr(ri, "timestamp", "")
        info.playerId = getattr(ri, "playerId", "bs-ai") or "bs-ai"
        info.playerName = getattr(ri, "playerName", "BS-AI") or "BS-AI"
        info.platform = getattr(ri, "platform", "steam") or "steam"
        info.trackingSystem = getattr(ri, "trackingSystem", "OculusTouch") or "OculusTouch"
        info.hmd = getattr(ri, "hmd", "OculusRift") or "OculusRift"
        info.controller = getattr(ri, "controller", "OculusTouch") or "OculusTouch"
        info.songHash = self.song_hash
        info.songName = self.song_name
        info.mapper = self.mapper
        info.difficulty = self.difficulty
        info.score = 0
        info.mode = self.mode
        info.environment = self.environment or "DefaultEnvironment"
        info.modifiers = ""
        info.jumpDistance = float(self.JD)
        info.leftHanded = False
        info.height = float(self.height)
        info.startTime = 0.0
        info.failTime = 0.0
        info.speed = 1.0

        G_np = G.detach().cpu().numpy() if torch.is_tensor(G) else np.asarray(G)
        frames = []
        for i in range(G_np.shape[0]):
            fr = Frame()
            fr.time = float(G_np[i, 0])
            fr.fps = int(round(self.fps))
            h = VRObject()
            h.x, h.y, h.z, h.x_rot, h.y_rot, h.z_rot, h.w_rot = [float(v) for v in G_np[i, 1:8]]
            l = VRObject()
            l.x, l.y, l.z, l.x_rot, l.y_rot, l.z_rot, l.w_rot = [float(v) for v in G_np[i, 8:15]]
            r = VRObject()
            r.x, r.y, r.z, r.x_rot, r.y_rot, r.z_rot, r.w_rot = [float(v) for v in G_np[i, 15:22]]
            fr.head, fr.left_hand, fr.right_hand = h, l, r
            frames.append(fr)

        b = Bsor()
        b.magic_number = 0x442D3D69
        b.file_version = 1
        b.info = info
        b.frames = frames
        b.notes = []
        b.walls = []
        b.heights = []
        b.pauses = []
        b.controller_offsets = None
        b.user_data = []
        with open(path, "wb") as f:
            b.write(f)

    def annotate(self, scorer_out, out_path):
        """Write per-note annotations CSV from a SaberScorer result (rec has n_idx)."""
        import csv
        notes = {i: n for i, n in enumerate(self.notes)}
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_idx", "time", "color", "cut_dir", "line_index", "line_layer",
                        "before", "after", "acc", "breakdown", "mult", "path"])
            for r in scorer_out.rec:
                ni = r.get("n_idx", -1)
                n = notes.get(ni, {})
                w.writerow([
                    ni,
                    "%.4f" % n.get("time", 0.0),
                    n.get("color", ""),
                    n.get("cut_dir", ""),
                    n.get("line_index", ""),
                    n.get("line_layer", ""),
                    "%.4f" % r["before"],
                    "%.4f" % r["after"],
                    r["acc"],
                    " ".join(str(x) for x in r["breakdown"]),
                    r["mult"],
                    r["path"],
                ])

    def eval_summary(self, scorer_out):
        """Compact per-note stats for progress JSON."""
        rec = scorer_out.rec
        n = len(rec)
        hits = sum(1 for r in rec if r["acc"] > 0)
        acc_avg = float(np.mean([r["acc"] for r in rec])) if n else 0.0
        before_avg = float(np.mean([r["before"] for r in rec])) if n else 0.0
        after_avg = float(np.mean([r["after"] for r in rec])) if n else 0.0
        return {
            "notes": n, "hit": hits, "miss": n - hits,
            "acc_avg": round(acc_avg, 3), "before_avg": round(before_avg, 4),
            "after_avg": round(after_avg, 4),
        }
