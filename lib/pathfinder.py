"""Rail pathfinder: deterministic swing planner for Beat Saber maps.

Pipeline (the "calculator"):
  Stage 0  MapGeometry    - parse map; compute note world positions/facings (same
                            math as torchsaber.note_geometry/_orientations, CPU)
                            and wall occupancy.
  Stage 1  SwingBudgets   - per-saber feasibility: between consecutive notes the
                            chosen swing directions must be connectable within
                            the time gap at the measured swing rate R.
  Stage 2  DP             - per-note cut-direction assignment (8 compass dots)
                            minimizing total swing travel subject to budgets;
                            soft preference to cut along each note's arrow.
  Stage 3  RailSynth      - tip/base/head rails -> 90 FPS frame transforms
                            (head dodges walls / ducks; tips carve notes along
                            the chosen direction; saber axis edge-on to swing).
  Stage 4  SelfCheck      - re-score the generated trajectory with the real
                            SaberScorer on CPU; report per-note before/after/acc
                            so failures are visible and the pipeline can iterate.
  Stage 5  Temperature    - softmax over dot directions + bounded perturbation
                            + trust-region rejection (post-prototype).

Coordinate conventions (match torchsaber): x = horizontal, y = up, z = forward
(player faces +z, notes fly in from +z). Frame rows = [t, head(7), left(7), right(7)].
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from bsdata import create_map
from torchsaber import (
    RANDOM_ROTATIONS, SaberScorer, q_conj, q_from_euler, q_mul, q_norm, q_rot,
    q_to_euler, v_cross, v_dot, v_norm,
)

DEG_TO_RAD = 0.0174532924
RAD_TO_DEG = 57.29578
SABER_LENGTH = 1.0
FPS = 90.0
# planning swing-rate budget (deg/s). measured human p99 ~2833, p50 ~468; use a
# comfortably attainable budget so the rails stay physical.
SWING_RATE = 2000.0
# per-note swing credit requirements (deg) used by Stage 1 budgets
BEFORE_NEED = 100.0
AFTER_NEED = 60.0
# keep consecutive swing windows from coinciding (see rotation-keyframe note)
WINDOW_MARGIN = 0.03
# same-hand notes closer than this (a double slash) share one swing window
CLUSTER_GAP = 0.06
# 8 discrete compass dot directions (horizontal tip-travel tangents), deg
DOT_ANGLES = np.arange(0, 360, 45, dtype=np.float64)


def _dot_dir(angle_deg):
    a = angle_deg * DEG_TO_RAD
    return np.array([math.cos(a), 0.0, math.sin(a)])


DOT_DIRS = np.stack([_dot_dir(a) for a in DOT_ANGLES])  # (8, 3)


# ---------------------------------------------------------------------------
# Stage 0: map geometry
# ---------------------------------------------------------------------------

class MapGeometry:
    """Notes/bombs/walls in world space, computed with scorer-exact math.

    Note hit position/facing are computed for a nominal head at (0, height, 0),
    using the exact torchsaber.note_geometry / note_orientations formulas on a
    90 FPS frame grid. The scorer recomputes them from whatever head trajectory
    we produce; keeping the head near (0, height, 0) keeps them consistent.
    """

    def __init__(self, map_dir, mode, difficulty, height=1.8, fps=FPS):
        self.map_dir = map_dir
        self.mode = mode
        self.difficulty = difficulty
        m = create_map(map_dir)
        bm = m.beatMaps[mode][difficulty]
        self.bpm = float(m.beatsPerMinute)
        self.NJS = float(bm.noteJumpMovementSpeed)
        self.JD = self.NJS          # map-only convention (1s reaction window)
        self.height = float(height)
        self.fps = float(fps)
        b2s = lambda b: b * 60.0 / self.bpm

        self.notes = []             # type 0/1
        self.bombs = []             # type 3
        for n in bm.notes:
            if n.type in (0, 1):
                self.notes.append({
                    "time": b2s(n.time), "color": n.type, "cut_dir": n.cutDirection,
                    "line_index": n.lineIndex, "line_layer": n.lineLayer,
                })
            elif n.type == 3:
                self.bombs.append({
                    "time": b2s(n.time), "line_index": n.lineIndex,
                    "line_layer": n.lineLayer,
                })
        self.notes.sort(key=lambda n: n["time"])
        self.bombs.sort(key=lambda b: b["time"])

        # walls in world coords
        self.walls = []
        for w in bm.obstacles:
            if w.type == 0:
                y0, y1 = 0.0, 5.0
            elif w.type == 1:
                y0, y1 = 2.0, 5.0
            else:
                continue
            x = _wall_x(w.lineIndex)
            self.walls.append({
                "t0": b2s(w.time), "t1": b2s(w.time + w.duration),
                "x0": x, "x1": x + w.width * 0.6, "y0": y0, "y1": y1,
            })
        self.walls.sort(key=lambda w: w["t0"])

        self._compute_note_positions()

    def _compute_note_positions(self):
        """Exact scorer geometry for all notes at their hit frames."""
        dev = "cpu"
        N = len(self.notes)
        if N == 0:
            self.note_time = np.zeros(0)
            self.pos = np.zeros((0, 3))
            self.fwd = np.zeros((0, 3))
            self.rot = np.zeros((0, 4))
            self.r_e = np.zeros(0, dtype=np.int64)
            self.t = np.zeros(0)
            return
        note_time = np.array([n["time"] for n in self.notes], dtype=np.float64)
        line_index = np.array([n["line_index"] for n in self.notes], dtype=np.int64)
        line_layer = np.array([n["line_layer"] for n in self.notes], dtype=np.int64)
        cut_dir = np.array([n["cut_dir"] for n in self.notes], dtype=np.int64)

        t_end = float(note_time[-1]) + self.JD / self.NJS + 1.0
        t = np.arange(0.0, t_end, 1.0 / self.fps)
        F = len(t)
        head = np.zeros((F, 3), dtype=np.float64)
        head[:, 1] = self.height
        head[:, 2] = 0.0

        D = {
            "t": torch.from_numpy(t), "head": torch.from_numpy(head),
            "note_time": torch.from_numpy(note_time),
            "line_index": torch.from_numpy(line_index),
            "line_layer": torch.from_numpy(line_layer),
            "ntype": np.zeros(N, dtype=np.int64),
            "cut_dir": torch.from_numpy(cut_dir),
            "NJS": self.NJS, "JD": self.JD, "height": self.height,
        }
        scorer = SaberScorer(dev)
        nc = scorer.note_geometry(D)
        r_e = np.searchsorted(t, note_time, side="right")
        e_idx = torch.arange(N)
        pos, rot, fwd = scorer.note_orientations(
            D, nc, e_idx, e_idx, torch.from_numpy(r_e))

        self.note_time = note_time
        self.pos = pos.numpy()
        self.rot = rot.numpy()
        self.fwd = fwd.numpy()
        self.r_e = r_e
        self.t = t

    # -- per-note world targets -------------------------------------------
    def note_targets(self, i):
        """(time, world pos, fwd unit vec) for note i."""
        return self.note_time[i], self.pos[i], self.fwd[i]

    def wall_at(self, t, x, y):
        """True if head point (x, y) at time t is inside a wall."""
        for w in self.walls:
            if w["t0"] <= t <= w["t1"]:
                if w["x0"] <= x <= w["x1"] and w["y0"] <= y <= w["y1"]:
                    return True
        return False


def _wall_x(line_index):
    return (-(4 - 1) * 0.5 + line_index) * 0.6


# ---------------------------------------------------------------------------
# Stage 1+2: per-saber swing DP over 8 discrete directions
# ---------------------------------------------------------------------------

def hand_notes(geo):
    """Split notes into per-hand (color) lists preserving time order."""
    left, right = [], []
    for i, n in enumerate(geo.notes):
        (left if n["color"] == 0 else right).append(i)
    return left, right


def _ang_between(a, b):
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.degrees(math.acos(c))


def plan_hand(geo, note_idx, rate=SWING_RATE):
    """DP over a hand's notes -> chosen dot direction index per note.

    Returns dict with per-note: dir_index (0..7), tangent (3,), cost info.
    Feasibility: angle(d_prev, d_next) <= rate * dt  (dt = t_next - t_prev).
    Objective: minimize total angular travel; soft preference to cut along the
    note's arrow (dot of tangent with note fwd).
    """
    K = 8
    m = len(note_idx)
    if m == 0:
        return {"dir": np.zeros(0, dtype=np.int64), "tangent": np.zeros((0, 3)),
                "feasible": np.zeros(0, dtype=bool), "cost": np.zeros(0)}
    ts = np.array([geo.note_time[i] for i in note_idx])
    fwds = np.stack([geo.fwd[i] for i in note_idx])
    # horizontal projection of each arrow -> preferred dot angle
    pref = []
    for f in fwds:
        h = np.array([f[0], 0.0, f[2]])
        n = np.linalg.norm(h)
        if n < 1e-6:
            pref.append(0)          # vertical arrow: no horizontal preference
        else:
            h = h / n
            # angle of h relative to +x axis in x-z plane (toward +z = forward)
            ang = math.degrees(math.atan2(h[2], h[0]))
            pref.append(int(round(ang / 45.0)) % 8)
    pref = np.array(pref)

    # per-gap feasibility between dot directions
    cost = np.zeros((m, K, K))       # cost[i-1->i] travel (deg)
    feasible = np.zeros((m, K, K), dtype=bool)
    for j in range(1, m):
        dt = ts[j] - ts[j - 1]
        for a in range(K):
            for b in range(K):
                ang = _ang_between(DOT_DIRS[a], DOT_DIRS[b])
                cost[j, a, b] = ang
                feasible[j, a, b] = ang <= rate * dt + 1e-9

    # Viterbi: maximize reward = -travel_cost - arrow_mismatch_penalty
    match_pen = np.zeros((m, K))
    for j in range(m):
        for k in range(K):
            ang = _ang_between(DOT_DIRS[k], fwds[j])
            # +45 deg if horizontal; vertical arrows penalized lightly
            match_pen[j, k] = ang * 0.25
    if m == 1:
        k = int(np.argmin(match_pen[0]))
        return {"dir": np.array([k]), "tangent": DOT_DIRS[k][None],
                "feasible": np.array([True]), "cost": np.array([0.0])}

    BIG = 1e9
    V = np.full((m, K), BIG)
    back = np.zeros((m, K), dtype=np.int64)
    V[0] = match_pen[0]
    for j in range(1, m):
        for b in range(K):
            cand = V[j - 1] + np.where(feasible[j, :, b], cost[j, :, b], BIG)
            k0 = int(np.argmin(cand))
            V[j, b] = cand[k0] + match_pen[j, b]
            back[j, b] = k0
    k = int(np.argmin(V[m - 1]))
    dirs = np.zeros(m, dtype=np.int64)
    for j in range(m - 1, -1, -1):
        dirs[j] = k
        if j > 0:
            k = back[j, k]
    tangents = DOT_DIRS[dirs]
    # per-gap feasible flag (best path's gaps)
    gap_ok = np.zeros(m, dtype=bool)
    gap_ok[0] = True
    for j in range(1, m):
        gap_ok[j] = feasible[j, dirs[j - 1], dirs[j]]
    return {"dir": dirs, "tangent": tangents, "feasible": gap_ok,
            "cost": np.array([V[j, dirs[j]] for j in range(m)])}


# ---------------------------------------------------------------------------
# Stage 3: rail synthesis -> 90 FPS frame transforms
# ---------------------------------------------------------------------------

def _quat_from_axis(n_hat):
    """Unit quaternion with q_rot(z, q) == n_hat (z = (0,0,1)). Minimal twist."""
    z = np.array([0.0, 0.0, 1.0])
    c = np.clip(np.dot(z, n_hat), -1.0, 1.0)
    if c > 0.99999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if c < -0.99999:
        return np.array([0.0, 1.0, 0.0, 0.0])  # 180 about x
    axis = np.cross(z, n_hat)
    axis = axis / np.linalg.norm(axis)
    ang = math.acos(c)
    return np.concatenate([axis * math.sin(ang / 2.0), [math.cos(ang / 2.0)]])


def _swing_basis(tangent):
    """Return (rot_axis, n_hat) for a swing with given tip tangent.

    Swing plane is vertical (rotation axis horizontal, perpendicular to the
    tangent and to up); the saber axis is edge-on to the motion.
    """
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(tangent, up)
    n_axis = np.linalg.norm(axis)
    if n_axis < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])   # vertical tangent: fallback
    else:
        axis = axis / n_axis
    n_hat = np.cross(tangent, axis)
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
    return axis, n_hat


def synth_hand(geo, plan, t, fps=FPS, center=None):
    """Build (hilt, tip, quat) 3D arrays for one hand over the frame grid.

    TIP rail: the tip passes exactly through each note position at its hit
    frame. SABER AXIS rail: the saber axis rotates through a big arc per note
    (BEFORE_NEED deg into the note, AFTER_NEED out) so the scorer accumulates
    swing credit; hilt = tip - L*n keeps |tip - hilt| = L everywhere.
    """
    idx = plan["note_idx"]
    m = len(idx)
    F = len(t)
    n_hat = np.zeros((F, 3))
    q = np.zeros((F, 4))
    q[:, 3] = 1.0

    if m == 0:
        return {"tip": np.zeros((F, 3)), "hilt": np.zeros((F, 3)),
                "quat": q, "n": n_hat}

    ts = np.array([geo.note_time[i] for i in idx]).copy()
    ps = np.stack([geo.pos[i] for i in idx])
    tangents = plan["tangent"]

    # stagger near-simultaneous same-hand notes so the spline can pass through
    # both (a diagonal slash through a stack)
    for j in range(1, m):
        if ts[j] - ts[j - 1] < 0.03:
            ts[j] = ts[j - 1] + 0.03

    # comfortable wrist center for this hand (near body, slightly in front)
    if center is None:
        center = np.array([0.0, 1.1, 0.3])
    center = np.asarray(center, dtype=np.float64)

    # ---- per-note swing geometry -------------------------------------------
    # for each note: saber axis n_j (edge-on: perpendicular to the tip tangent,
    # pointing roughly from the wrist toward the note), rotation axis a_j,
    # wrist H_j = P_j - L*n_j (so tip = P_j at the hit).
    n_j = np.zeros((m, 3))
    a_j = np.zeros((m, 3))
    H_j = np.zeros((m, 3))
    q_hit = np.zeros((m, 4))
    for j in range(m):
        d = tangents[j]
        r = ps[j] - center
        # perpendicular part of r w.r.t. the tangent (edge-on, toward note)
        n = r - np.dot(r, d) * d
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            n = np.cross(d, np.array([0.0, 1.0, 0.0]))
            nn = np.linalg.norm(n)
        n = n / nn
        n_j[j] = n
        a = np.cross(d, n)
        an = np.linalg.norm(a)
        if an < 1e-6:
            a = np.cross(d, np.array([0.0, 0.0, 1.0]))
            an = np.linalg.norm(a)
        a = a / an
        a_j[j] = a
        H_j[j] = ps[j] - SABER_LENGTH * n
        q_hit[j] = _quat_from_axis(n)

    # ---- rotation keyframes -------------------------------------------------
    # each note contributes an entry / hit / exit keyframe. half-window shrinks
    # when notes are dense so consecutive swings do not overlap in time.
    # A margin keeps exit_j strictly before entry_{j+1}: if they coincide, the
    # slerp schedule would jump from note j's hit straight toward note j+1's
    # entry, reversing the saber rotation at the crossing and flipping the
    # swing-plane normal (which zeroes the S0 before-credit).
    # Near-simultaneous same-hand notes (a double slash) are clustered into a
    # single swing event with one entry / one exit and a hit keyframe per note,
    # so the slerp rotates forward between hits instead of reversing through an
    # interleaved entry/exit pair.
    key_t = []
    key_q = []
    key_Ht = []
    key_H = []
    i = 0
    while i < m:
        j = i
        while j + 1 < m and ts[j + 1] - ts[j] < CLUSTER_GAP:
            j += 1
        w_in = 0.25
        if i > 0:
            w_in = min(w_in, (ts[i] - ts[i - 1]) / 2.0 - WINDOW_MARGIN)
        w_out = 0.25
        if j < m - 1:
            w_out = min(w_out, (ts[j + 1] - ts[j]) / 2.0 - WINDOW_MARGIN)
        w_in = max(w_in, 0.02)
        w_out = max(w_out, 0.02)
        entry = _rot_about(q_hit[i], a_j[i], -BEFORE_NEED)
        exitq = _rot_about(q_hit[j], a_j[j], AFTER_NEED)
        # rotation rail: entry -> one hit per note -> exit
        key_t += [ts[i] - w_in]
        key_q += [entry]
        for k in range(i, j + 1):
            key_t += [ts[k]]
            key_q += [q_hit[k]]
        key_t += [ts[j] + w_out]
        key_q += [exitq]
        # wrist rail: pinned at H_j[i] through the first note's crossing (which
        # the scorer registers at ~the hit frame), then transitions to H_j[j]
        # in time for the last note's hit so the tip still lands exactly on it.
        # A moving wrist at the crossing frame contaminates mid_prev - hilt and
        # flips the swing-plane normal => zeroed before-credit.
        key_Ht += [ts[i] - w_in]
        key_H += [H_j[i]]
        for k in range(i, j + 1):
            key_Ht += [ts[k]]
            key_H += [H_j[k]]
        key_Ht += [ts[j] + w_out]
        key_H += [H_j[j]]
        i = j + 1

    # quaternion rail: slerp through the rotation keyframes
    key_t = np.array(key_t)
    key_q = np.array(key_q)
    order = np.argsort(key_t)
    key_t, key_q = key_t[order], key_q[order]
    # wrist rail: independent timeline so the wrist holds still through each
    # note's crossing before moving on to the next wrist target.
    key_Ht = np.array(key_Ht)
    key_H = np.array(key_H)
    oH = np.argsort(key_Ht)
    key_Ht, key_H = key_Ht[oH], key_H[oH]
    hilt = _lerp_at(key_Ht, key_H, t)

    # saber axis rail: slerp through the rotation keyframes
    for f in range(F):
        q[f] = _slerp_schedule(key_t, key_q, t[f])
        n_hat[f] = q_rot_np(np.array([0.0, 0.0, 1.0]), q[f])

    # tip rail = hilt + L*n ; passes exactly through each note at its hit time
    # (hilt and quaternion keyframes are exact at ts[j] by construction), so the
    # crossing is registered by the scorer's forward scan and acc is perfect.
    tip = hilt + SABER_LENGTH * n_hat
    return {"tip": tip, "hilt": hilt, "quat": q, "n": n_hat}


def _rot_about(q, axis, deg):
    """Quaternion = q premultiplied by a rotation of deg about unit axis."""
    ang = deg * DEG_TO_RAD
    h = ang / 2.0
    dq = np.concatenate([axis * math.sin(h), [math.cos(h)]])
    return q_np_mul(dq, q)


def _lerp_at(knot_t, knot_v, t):
    out = np.zeros((len(t), knot_v.shape[1]))
    k = np.clip(np.searchsorted(knot_t, t, side="right") - 1, 0, len(knot_t) - 2)
    for f in range(len(t)):
        k0, k1 = k[f], k[f] + 1
        span = max(knot_t[k1] - knot_t[k0], 1e-9)
        u = np.clip((t[f] - knot_t[k0]) / span, 0.0, 1.0)
        out[f] = knot_v[k0] * (1 - u) + knot_v[k1] * u
    return out


def _slerp_schedule(knot_t, knot_q, tf):
    """Evaluate the piecewise slerp of (knot_t, knot_q) at time tf."""
    k = int(np.clip(np.searchsorted(knot_t, tf, side="right") - 1, 0, len(knot_t) - 2))
    t0, t1 = knot_t[k], knot_t[k + 1]
    span = max(t1 - t0, 1e-9)
    u = np.clip((tf - t0) / span, 0.0, 1.0)
    return q_slerp_np(knot_q[k], knot_q[k + 1], u)


def _hermite(knot_t, knot_p, knot_v, t):
    """Cubic Hermite spline through (knot_t, knot_p) with velocities knot_v."""
    knot_t = np.asarray(knot_t)
    knot_p = np.asarray(knot_p)
    n = len(knot_t)
    if n < 2:
        out = np.zeros((len(t), 3))
        out[:] = knot_p[0]
        return out
    k = np.clip(np.searchsorted(knot_t, t, side="right") - 1, 0, n - 2)
    out = np.zeros((len(t), 3))
    for f in range(len(t)):
        i = k[f]
        h = max(knot_t[i + 1] - knot_t[i], 1e-9)
        u = np.clip((t[f] - knot_t[i]) / h, 0.0, 1.0)
        u2, u3 = u * u, u * u * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        out[f] = (h00 * knot_p[i] + h10 * h * knot_v[i]
                  + h01 * knot_p[i + 1] + h11 * h * knot_v[i + 1])
    return out


def q_np_mul(a, b):
    aw, ax, ay, az = a[3], a[0], a[1], a[2]
    bw, bx, by, bz = b[3], b[0], b[1], b[2]
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def q_slerp_np(a, b, u):
    u = np.clip(u, 0.0, 1.0)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot < 0.0:
        b = -b
        dot = -dot
    theta = math.acos(dot)
    if theta < 1e-7:
        out = a + (b - a) * u
        return out / (np.linalg.norm(out) + 1e-12)
    sin_t = math.sin(theta)
    c0 = math.sin((1 - u) * theta) / sin_t
    c1 = math.sin(u * theta) / sin_t
    out = a * c0 + b * c1
    return out / (np.linalg.norm(out) + 1e-12)


def q_rot_np(v, q):
    """Rotate vectors v (...,3) by quaternions q (...,4)."""
    v = np.asarray(v, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    w = q[..., 3:4]
    xyz = q[..., :3]
    t = np.cross(xyz, v, axis=-1) + w * v
    return v + 2.0 * np.cross(xyz, t, axis=-1)


def _spline_at(knots, vals, t):
    """Piecewise-linear interpolation of vals at knots -> values on grid t."""
    out = np.zeros((len(t), vals.shape[1]))
    k = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
    for f in range(len(t)):
        k0, k1 = k[f], k[f] + 1
        span = max(knots[k1] - knots[k0], 1e-9)
        u = np.clip((t[f] - knots[k0]) / span, 0.0, 1.0)
        out[f] = vals[k0] * (1 - u) + vals[k1] * u
    return out


def synth_head(geo, t, hand_tips):
    """Head rail: keep out of walls (dodge full walls laterally, duck crouch)."""
    F = len(t)
    head = np.zeros((F, 3))
    head[:, 1] = geo.height
    head[:, 2] = 0.0
    # nominal standing x lanes available to the head (left/right/center of walls)
    free_x = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
    cur = 0.0
    for f in range(F):
        tf = t[f]
        blocked = [x for x in free_x if geo.wall_at(tf, x, geo.height)]
        cand = [x for x in free_x if x not in blocked]
        if not cand:
            cand = free_x
        # pick candidate nearest to current, with a small comfort bias to center
        best = min(cand, key=lambda x: abs(x - cur) + 0.15 * abs(x))
        cur += 0.3 * (best - cur)
        head[f, 0] = cur
    # duck under crouch walls: if a crouch wall covers current lane, lower head
    for f in range(F):
        tf = t[f]
        if any(w["y0"] <= geo.height <= w["y1"] and w["x0"] <= head[f, 0] <= w["x1"]
               and w["t0"] <= tf <= w["t1"] for w in geo.walls):
            head[f, 1] = min(head[f, 1], 1.4)
    return head


def build_frames(geo, left_plan, right_plan):
    """Assemble full 90 FPS frame array (F, 23): [t, head7, left7, right7]."""
    t = geo.t
    F = len(t)
    L = synth_hand(geo, left_plan, t)
    R = synth_hand(geo, right_plan, t)
    head = synth_head(geo, t, [L["tip"], R["tip"]])

    frames = np.zeros((F, 23), dtype=np.float32)
    frames[:, 0] = t.astype(np.float32)
    frames[:, 1:4] = head
    frames[:, 4:8] = [0, 0, 0, 1]
    frames[:, 8:11] = L["hilt"]
    frames[:, 11:15] = L["quat"]
    frames[:, 15:18] = R["hilt"]
    frames[:, 18:22] = R["quat"]
    frames[:, 22] = 0.0
    return frames


# ---------------------------------------------------------------------------
# Stage 4: self-consistency via the real SaberScorer on CPU
# ---------------------------------------------------------------------------

def self_check(geo, frames, progress_dir=None):
    """Score generated frames with the real scorer; return per-note report + total.

    The scorer's per-note accuracy uses the recorded cut point (where the saber
    actually crossed the note). For generated trajectories that point is not
    known a priori, so we compute it from the trajectory itself with the
    scorer's exact geometry (note orientations from our head rail, crossing
    interpolation between consecutive frames) and feed it back as cut_point.
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
    tipL = hL + q_rot_np(z, qL)
    tipR = hR + q_rot_np(z, qR)

    N = len(geo.notes)
    D = {
        "t": torch.from_numpy(t),
        "head": torch.from_numpy(head),
        "hiltL": torch.from_numpy(hL), "tipL": torch.from_numpy(tipL),
        "hiltR": torch.from_numpy(hR), "tipR": torch.from_numpy(tipR),
        "note_time": torch.from_numpy(geo.note_time),
        "line_index": torch.from_numpy(np.array([n["line_index"] for n in geo.notes], dtype=np.int64)),
        "line_layer": torch.from_numpy(np.array([n["line_layer"] for n in geo.notes], dtype=np.int64)),
        "ntype": torch.from_numpy(np.array([n["color"] for n in geo.notes], dtype=np.int64)),
        "cut_dir": torch.from_numpy(np.array([n["cut_dir"] for n in geo.notes], dtype=np.int64)),
        "e_idx": torch.arange(N),
        "n_idx": torch.arange(N),
        "r_e": torch.from_numpy(geo.r_e),
        "hand": torch.from_numpy(np.array([n["color"] for n in geo.notes], dtype=np.int64)),
        "cut_point": torch.zeros((N, 3)),
        "J": _calc_J(t),
        "NJS": geo.NJS, "bpm": geo.bpm, "JD": geo.JD, "height": geo.height,
    }
    scorer = SaberScorer("cpu")
    cut_points = _cut_points_np(scorer, D, tipL, tipR)
    D["cut_point"] = torch.from_numpy(cut_points)
    res = scorer(D)
    return res


def v_cross_np(a, b):
    return np.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], axis=-1)


def v_norm_np(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def v_angle_np(a, b):
    c = np.clip(np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12), -1, 1)
    return np.degrees(np.arccos(c))


def _cut_points_np(scorer, D, tipL, tipR):
    """Compute each note's crossing point on the note plane from the tip path.

    Mirrors the scorer's registration math in numpy: the tip crosses the plane
    normal np_ = cpn x fwd between consecutive frames; cut point is the linear
    interpolation of the crossing. Notes that never cross get a far point so
    their accuracy reads 0.
    """
    t = D["t"].numpy()
    F = len(t)
    N = int(D["e_idx"].shape[0])
    r_e = D["r_e"].numpy()
    hand = D["hand"].numpy()
    tips = [tipL, tipR]
    h_l = D["hiltL"].numpy(); h_r = D["hiltR"].numpy()
    hilts = [h_l, h_r]
    nc = scorer.note_geometry(D)
    pos, rot, fwd = scorer.note_orientations(
        D, nc, torch.arange(N), torch.arange(N), D["r_e"])
    pos = pos.numpy(); fwd = fwd.numpy()

    normals = np.zeros((2, F, 3))
    segs = np.zeros((2, F))
    for h in (0, 1):
        ax = tips[h] - hilts[h]
        ax_prev = np.concatenate([ax[:1], ax[:-1]])
        mid_prev = (hilts[h][:1] + ax[:1] / 1.0) * 0.0 + (
            (hilts[h] + tips[h]) / 2.0)
        mid_prev = np.concatenate([mid_prev[:1], mid_prev[:-1]])
        normal = v_norm_np(v_cross_np(ax, mid_prev - hilts[h]))
        seg = v_angle_np(ax, ax_prev)
        seg[0] = 0.0
        normals[h] = normal
        segs[h] = seg

    cp = np.full((N, 3), 1000.0)   # far away => acc 0 if never crossed
    for i in range(N):
        h = int(hand[i])
        f0 = int(r_e[i])
        if not (1 <= f0 < F):
            continue
        cpn = normals[h, f0]
        n_norm = np.linalg.norm(cpn)
        if n_norm < 1e-9:
            continue
        cpn = cpn / n_norm
        np_ = v_cross_np(cpn, fwd[i])
        # scan forward from r_e (like the scorer: crossing registered on the
        # first frame where the tip is on the other side, up to 0.4 s later)
        prev = tips[h][f0 - 1]
        s_prev = np.dot(np_, prev - pos[i])
        cut = None
        f_end = min(F, f0 + int(round(0.4 * 90.0)) + 2)
        for f in range(f0, f_end):
            cur = tips[h][f]
            s_curr = np.dot(np_, cur - pos[i])
            if s_prev == 0 or (s_prev > 0) != (s_curr > 0):
                denom = np.dot(np_, cur - prev)
                if abs(denom) > 1e-12:
                    dist_r = np.dot(np_, pos[i] - prev) / denom
                    cut = prev + dist_r * (cur - prev)
                elif s_prev == 0:
                    cut = prev
                break
            prev, s_prev = cur, s_curr
        if cut is not None:
            cp[i] = cut
    return cp


def _calc_J(t):
    F = len(t)
    J = 2
    for f in range(F):
        c = 0
        j = f - 1
        while j >= 0 and (t[f] - t[j + 1] < 0.4):
            c += 1
            j -= 1
        J = max(J, c + 2)
    return min(J, 501)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(map_dir, mode, difficulty, out_dir=None):
    geo = MapGeometry(map_dir, mode, difficulty)
    left, right = hand_notes(geo)
    lp = plan_hand(geo, left)
    rp = plan_hand(geo, right)
    lp["note_idx"] = left
    rp["note_idx"] = right
    frames = build_frames(geo, lp, rp)
    res = self_check(geo, frames)
    print("notes:", len(geo.notes), " bombs:", len(geo.bombs), " walls:", len(geo.walls))
    print("total score:", res.total)
    n_hit = sum(1 for r in res.rec if r["acc"] > 0)
    print("hits:", n_hit, "/", len(res.rec))
    print("avg before %.3f after %.3f acc %.3f" % (
        np.mean([r["before"] for r in res.rec]),
        np.mean([r["after"] for r in res.rec]),
        np.mean([r["acc"] for r in res.rec])))
    # per-note breakdown
    for r in res.rec:
        n = geo.notes[r["n_idx"]]
        print("  t=%7.3f c=%d cd=%d before=%6.3f after=%6.3f acc=%3d  -> %3d x%d" % (
            n["time"], n["color"], n["cut_dir"], r["before"], r["after"], r["acc"],
            int(round(r["before"] * 70) + round(r["after"] * 30) + r["acc"]), r["mult"]))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "frames.npy"), frames)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({"total": res.total, "notes": len(geo.notes),
                       "hits": n_hit}, f, indent=2)
    return geo, frames, res


def debug_misses(geo, frames):
    """Print why notes miss: tip-plane distance at r_e-1/r_e and sign flip."""
    t = frames[:, 0].astype(np.float64)
    F = frames.shape[0]
    head = frames[:, 1:4].astype(np.float64)
    def _tip(hilt_col, quat_col):
        h = frames[:, hilt_col:hilt_col + 3].astype(np.float64)
        q = frames[:, quat_col:quat_col + 4].astype(np.float64)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        z = np.zeros((F, 3)); z[:, 2] = 1.0
        return h + q_rot_np(z, q), h
    tipL, hiltL = _tip(8, 11)
    tipR, hiltR = _tip(15, 18)
    D = {
        "t": torch.from_numpy(t), "head": torch.from_numpy(head),
        "hiltL": torch.from_numpy(hiltL), "tipL": torch.from_numpy(tipL),
        "hiltR": torch.from_numpy(hiltR), "tipR": torch.from_numpy(tipR),
        "note_time": torch.from_numpy(geo.note_time),
        "line_index": torch.from_numpy(np.array([n["line_index"] for n in geo.notes], dtype=np.int64)),
        "line_layer": torch.from_numpy(np.array([n["line_layer"] for n in geo.notes], dtype=np.int64)),
        "ntype": torch.from_numpy(np.array([n["color"] for n in geo.notes], dtype=np.int64)),
        "cut_dir": torch.from_numpy(np.array([n["cut_dir"] for n in geo.notes], dtype=np.int64)),
        "e_idx": torch.arange(len(geo.notes)), "n_idx": torch.arange(len(geo.notes)),
        "r_e": torch.from_numpy(geo.r_e),
        "hand": torch.from_numpy(np.array([n["color"] for n in geo.notes], dtype=np.int64)),
        "J": _calc_J(t), "NJS": geo.NJS, "bpm": geo.bpm, "JD": geo.JD, "height": geo.height,
    }
    scorer = SaberScorer("cpu")
    nc = scorer.note_geometry(D)
    N = len(geo.notes)
    pos, _, fwd = scorer.note_orientations(
        D, nc, torch.arange(N), torch.arange(N), D["r_e"])
    pos = pos.numpy(); fwd = fwd.numpy()
    tips = [tipL, tipR]; hilts = [hiltL, hiltR]
    n_miss = 0
    for i in range(N):
        r_e = int(geo.r_e[i])
        if not (1 <= r_e < F):
            continue
        h = int(D["hand"][i])
        np_ = v_cross_np(v_cross_np(tips[h][r_e] - hilts[h][r_e],
                                   (hilts[h][r_e] + tips[h][r_e]) / 2.0 - hilts[h][r_e]),
                        fwd[i])
        nnp = np.linalg.norm(np_)
        if nnp < 1e-9:
            continue
        np_ = np_ / nnp
        s_p = np.sign(np.dot(np_, tips[h][r_e - 1] - pos[i]))
        s_c = np.sign(np.dot(np_, tips[h][r_e] - pos[i]))
        d_c = np.linalg.norm(tips[h][r_e] - pos[i])
        if s_p == s_c or s_c == 0.0 and s_p == 0.0:
            n_miss += 1
            if n_miss <= 12:
                print("MISS t=%.3f re=%d hand=%d s_p=%+d s_c=%+d d_c=%.3f d_prev=%.3f"
                      % (geo.note_time[i], r_e, h, s_p, s_c, d_c,
                         np.linalg.norm(tips[h][r_e - 1] - pos[i])))
    print("misses (no sign flip at r_e):", n_miss, "/", N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--mode", default="Standard")
    ap.add_argument("--difficulty", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.map_dir, args.mode, args.difficulty, args.out)


if __name__ == "__main__":
    main()