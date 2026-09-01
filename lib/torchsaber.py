"""torchsaber: a differentiable torch/CUDA port of SimSaber scoring.

Faithfully replicates the reference at /tmp/opencode/simsaber (including the
ScoreManager in-place list-iteration semantics) but computes all rating math in
torch float64 so gradients flow through saber/head trajectories.

Verified against the real-driver reference on 4 fixtures to <1e-9 per-note
rating diff (see verify_torchsaber.py).

Typical use:
    D = torchsaber.preprocess(map_dir, replay_path)   # numpy arrays (CPU)
    TD = torchsaber.to_tensor(D, device)              # torch tensors
    scorer = SaberScorer(device)
    res = scorer(TD)          # res.before/.after/.acc_float differentiable
    res.total                # integer score matching the game scorer
"""
import numpy as np
import torch

DEG_TO_RAD = 0.0174532924
PI = 3.14159274
RAD_TO_DEG = 57.29578

RANDOM_ROTATIONS = torch.tensor([
    [-0.9543871, -0.1183784, 0.2741019],
    [0.7680854, -0.08805521, 0.6342642],
    [-0.6780157, 0.306681, -0.6680131],
    [0.1255014, 0.9398643, 0.3176546],
    [0.365105, -0.3664974, -0.8557909],
    [-0.8790653, -0.06244748, -0.4725934],
    [0.01886305, -0.8065798, 0.5908241],
    [-0.1455435, 0.8901445, 0.4318099],
    [0.07651193, 0.9474725, -0.3105508],
    [0.1306983, -0.2508438, -0.9591639],
], dtype=torch.float64)


# ---------------------------------------------------------------------------
# vector / quaternion helpers (float64, Unity/SimSaber semantics)
# ---------------------------------------------------------------------------

def v_cross(a, b):
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], dim=-1)


def v_dot(a, b):
    return (a * b).sum(-1)


def v_norm(v):
    mag = torch.sqrt(v_dot(v, v) + 1e-30)
    safe = torch.where(mag > 1e-5, mag, torch.ones_like(mag))
    out = v / safe[..., None]
    return torch.where((mag > 1e-5)[..., None], out, torch.zeros_like(out))


def v_angle(a, b):
    mag2 = v_dot(a, a) * v_dot(b, b)
    denom = torch.sqrt(mag2)
    tiny = denom < 1e-15
    safe = torch.where(tiny, torch.ones_like(denom), torch.sqrt(mag2 + 1e-30))
    dot = v_dot(a, b) / safe
    cross = v_cross(a, b)
    cmag = torch.sqrt(v_dot(cross, cross) + 1e-30) / safe
    ang = torch.atan2(cmag, dot) * RAD_TO_DEG
    return torch.where(tiny, torch.zeros_like(ang), ang)


def q_mul(a, b):
    aw, ax, ay, az = a[..., 3], a[..., 0], a[..., 1], a[..., 2]
    bw, bx, by, bz = b[..., 3], b[..., 0], b[..., 1], b[..., 2]
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def q_conj(q):
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)


def q_norm(q):
    return q / torch.sqrt(torch.clamp(v_dot(q, q), 1e-300, None))[..., None]


def q_rot(v, q):
    vq = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
    return q_mul(q_mul(q, vq), q_conj(q))[..., :3]


def q_from_euler(yaw, pitch, roll):
    yaw = yaw * DEG_TO_RAD
    pitch = pitch * DEG_TO_RAD
    roll = roll * DEG_TO_RAD
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    return torch.stack([
        sy * cp * cr + cy * sp * sr,
        cy * sp * cr - sy * cp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * cp * cr + sy * sp * sr,
    ], dim=-1)


def q_to_euler(q):
    unit = v_dot(q, q)
    test = q[..., 0] * q[..., 3] - q[..., 1] * q[..., 2]
    x = torch.full_like(test, PI / 2)
    y = torch.where(test > 0.4995 * unit, 2 * torch.atan2(q[..., 1], q[..., 0]),
                    torch.zeros_like(test))
    y = torch.where(test < -0.4995 * unit, -2 * torch.atan2(q[..., 1], q[..., 0]), y)
    z = torch.zeros_like(test)
    m = (test <= 0.4995 * unit) & (test >= -0.4995 * unit)
    x = torch.where(m, torch.asin(2 * (q[..., 3] * q[..., 0] - q[..., 1] * q[..., 2])), x)
    y = torch.where(m, torch.atan2(2 * q[..., 3] * q[..., 1] + 2 * q[..., 2] * q[..., 0],
                                   1 - 2 * (q[..., 0] ** 2 + q[..., 1] ** 2)), y)
    z = torch.where(m, torch.atan2(2 * q[..., 3] * q[..., 2] + 2 * q[..., 0] * q[..., 1],
                                   1 - 2 * (q[..., 2] ** 2 + q[..., 0] ** 2)), z)
    return torch.remainder(torch.stack([x, y, z], dim=-1) * RAD_TO_DEG, 360.0)


def q_slerp(a, b, u):
    u = torch.clamp(u, 0.0, 1.0)
    theta = torch.acos(torch.clamp(v_dot(a, b), -1.0, 1.0))
    tiny = theta < 1e-7
    safe = torch.where(tiny, torch.ones_like(theta), theta)
    sin_t = torch.sin(safe)
    c0 = torch.where(tiny, 1.0 - u, torch.sin((1 - u) * safe) / sin_t)
    c1 = torch.where(tiny, u, torch.sin(u * safe) / sin_t)
    out = a * c0[..., None] + b * c1[..., None]
    return q_norm(out)


def q_lerp(a, b, u):
    u = torch.clamp(u, 0.0, 1.0)
    out = a + (b - a) * u[..., None]
    return q_norm(out)


def look_rotation(forward, up):
    up = torch.broadcast_to(up, forward.shape)
    x_axis = v_norm(v_cross(up, forward))
    y_axis = v_norm(v_cross(forward, x_axis))
    z_axis = v_norm(forward)
    M00, M01, M02 = x_axis[..., 0], y_axis[..., 0], z_axis[..., 0]
    M10, M11, M12 = x_axis[..., 1], y_axis[..., 1], z_axis[..., 1]
    M20, M21, M22 = x_axis[..., 2], y_axis[..., 2], z_axis[..., 2]
    X = 1 + M00 + M11 + M22
    w = torch.sqrt(X + 1e-30) / 2
    safe_w = torch.where(X <= 0, torch.ones_like(w), w)
    x = (M21 - M12) / (4 * safe_w)
    y = (M02 - M20) / (4 * safe_w)
    z = (M10 - M01) / (4 * safe_w)
    return q_norm(torch.stack([x, y, z, w], dim=-1))


def _lerp_unclamped(a, b, t):
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=a.dtype, device=a.device)
    return a + (b - a) * t


def _lerp(a, b, t):
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=a.dtype, device=a.device)
    t = torch.clamp(t, 0.0, 1.0)
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# preprocessing (numpy, CPU) -- data loading, not differentiable
# ---------------------------------------------------------------------------

def preprocess(map_dir, replay_path):
    from bsor.Bsor import make_bsor
    from interpretMapFiles import create_map

    m = create_map(map_dir)
    with open(replay_path, "rb") as f:
        r = make_bsor(f)
    beatmap = m.beatMaps[r.info.mode][r.info.difficulty]
    NJS = float(beatmap.noteJumpMovementSpeed)
    bpm = float(m.beatsPerMinute)
    JD = float(r.info.jumpDistance)
    height = float(r.info.height)

    frames = r.frames[1:]
    F = len(frames)
    t = np.array([fr.time for fr in frames], dtype=np.float64)
    head = np.array([[fr.head.x, fr.head.y, fr.head.z] for fr in frames], dtype=np.float64)

    hiltL = np.array([[fr.left_hand.x, fr.left_hand.y, fr.left_hand.z] for fr in frames], dtype=np.float64)
    hiltR = np.array([[fr.right_hand.x, fr.right_hand.y, fr.right_hand.z] for fr in frames], dtype=np.float64)
    qL = np.array([[fr.left_hand.x_rot, fr.left_hand.y_rot, fr.left_hand.z_rot, fr.left_hand.w_rot] for fr in frames], dtype=np.float64)
    qR = np.array([[fr.right_hand.x_rot, fr.right_hand.y_rot, fr.right_hand.z_rot, fr.right_hand.w_rot] for fr in frames], dtype=np.float64)

    z = np.zeros((F, 3))
    z[:, 2] = 1.0
    qL_t = torch.from_numpy(qL)
    qR_t = torch.from_numpy(qR)
    z_t = torch.from_numpy(z)
    tipL = hiltL + q_rot(z_t, qL_t).numpy()
    tipR = hiltR + q_rot(z_t, qR_t).numpy()

    notes = beatmap.notes
    note_time = np.array([n.time * 60.0 / bpm for n in notes], dtype=np.float64)
    line_index = np.array([n.lineIndex for n in notes], dtype=np.int64)
    line_layer = np.array([n.lineLayer for n in notes], dtype=np.int64)
    ntype = np.array([n.type for n in notes], dtype=np.int64)
    cut_dir = np.array([n.cutDirection for n in notes], dtype=np.int64)
    note_id = 30000 + line_index * 1000 + line_layer * 100 + ntype * 10 + cut_dir

    ev_note_id = np.array([n.note_id for n in r.notes], dtype=np.int64)
    ev_time = np.array([n.event_time for n in r.notes], dtype=np.float64)
    ev_saber = np.array([n.cut.saberType for n in r.notes], dtype=np.int64)
    ev_cutpoint = np.array([n.cut.cutPoint for n in r.notes], dtype=np.float64)
    order = np.argsort(ev_time, kind="stable")
    ev_note_id, ev_time, ev_saber, ev_cutpoint = ev_note_id[order], ev_time[order], ev_saber[order], ev_cutpoint[order]

    # event -> note matching (FIFO per id among spawned, uncut notes)
    r_e_all = np.searchsorted(t, ev_time, side="right")
    spawn_ahead = 1 + JD / NJS * 0.5
    spawn_time = note_time - spawn_ahead
    spawn_frame = np.searchsorted(t, spawn_time, side="left")
    curs = {}
    ev_note = np.full(len(ev_time), -1, dtype=np.int64)
    for e in range(len(ev_time)):
        q = curs.setdefault(ev_note_id[e], 0)
        # find next note with this id not yet consumed
        nxt = -1
        for i in range(q, len(notes)):
            if note_id[i] == ev_note_id[e]:
                nxt = i
                break
        if nxt == -1:
            continue
        curs[ev_note_id[e]] = nxt + 1
        if spawn_frame[nxt] < r_e_all[e] and r_e_all[e] < F:
            ev_note[e] = nxt

    ok = ev_note >= 0
    e_idx = np.where(ok)[0]
    n_idx = ev_note[e_idx]
    r_e = r_e_all[e_idx]
    hand = ev_saber[e_idx]
    cut_point = ev_cutpoint[e_idx]

    # swing window length J (max frames in [t-0.4, t] window, + margin)
    J = 2
    for f in range(F):
        c = 0
        j = f - 1
        while j >= 0 and (t[f] - t[j + 1] < 0.4):
            c += 1
            j -= 1
        J = max(J, c + 2)
    J = min(J, 501)

    return {
        "t": t, "head": head,
        "hiltL": hiltL, "tipL": tipL, "hiltR": hiltR, "tipR": tipR,
        "note_time": note_time, "line_index": line_index, "line_layer": line_layer,
        "ntype": ntype, "cut_dir": cut_dir, "note_id": note_id,
        "e_idx": e_idx, "n_idx": n_idx, "r_e": r_e, "hand": hand,
        "cut_point": cut_point,
        "NJS": NJS, "bpm": bpm, "JD": JD, "height": height,
        "J": J,
        "recorded": float(r.info.score),
    }


def to_tensor(D, device):
    keys = ["t", "head", "hiltL", "tipL", "hiltR", "tipR",
            "note_time", "line_index", "line_layer", "ntype", "cut_dir",
            "e_idx", "n_idx", "r_e", "hand", "cut_point"]
    out = {}
    for k in keys:
        out[k] = torch.as_tensor(D[k], device=device)
    if D["cut_dir"].dtype == np.int64:
        out["cut_dir"] = out["cut_dir"].to(torch.int64)
    out["J"] = D["J"]
    out["NJS"] = float(D["NJS"])
    out["bpm"] = float(D["bpm"])
    out["JD"] = float(D["JD"])
    out["height"] = float(D["height"])
    return out


# ---------------------------------------------------------------------------
# scorer module
# ---------------------------------------------------------------------------

class SaberScorer(torch.nn.Module):
    def __init__(self, device=None, dtype=torch.float64):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    # ---- stage 1: saber arrays -------------------------------------------------
    def saber_arrays(self, D, hilt, tip):
        axis = tip - hilt
        axis_prev = torch.cat([axis[:1], axis[:-1]], dim=0)
        hilt_prev = torch.cat([hilt[:1], hilt[:-1]], dim=0)
        tip_prev = torch.cat([tip[:1], tip[:-1]], dim=0)
        mid_prev = (hilt_prev + tip_prev) / 2.0
        normal = v_norm(v_cross(axis, mid_prev - hilt))
        seg = v_angle(axis, axis_prev)
        seg = seg.clone()
        seg[0] = 0.0
        return normal, seg

    # ---- stage 2: swing precompute (S0, j_stop) ---------------------------------
    def swing_precompute(self, normal, seg, t, J):
        F = t.shape[0]
        S0 = torch.zeros(F, device=t.device, dtype=t.dtype)
        j_stop = torch.full((F,), -1, dtype=torch.int64, device=t.device)
        arange = torch.arange(F, device=t.device, dtype=torch.int64)
        for o in range(1, J):
            j = arange - o
            valid = j >= 0
            jc = torch.clamp(j, 0, F - 1)
            tc = torch.clamp(j + 1, 0, F - 1)
            ang = v_angle(normal, normal[jc])
            time_fail = t - t[tc] >= 0.4
            ang_fail = ang >= 90.0
            fail = (~valid) | time_fail | ang_fail
            bad = torch.where(fail, j, torch.tensor(-1, dtype=torch.int64, device=t.device))
            j_stop = torch.maximum(j_stop, bad)
        wsum = torch.zeros(F, device=t.device, dtype=t.dtype)
        for o in range(1, J):
            j = arange - o
            jc = torch.clamp(j, 0, F - 1)
            included = j > j_stop
            ang = v_angle(normal, normal[jc])
            w = torch.where(ang < 75.0, torch.ones_like(ang), (90.0 - ang) / 15.0)
            wsum = wsum + torch.where(included, seg[jc] * w, torch.zeros_like(seg))
        S0 = wsum / 100.0
        return S0, j_stop

    def swing_batch(self, S0v, jS, segv, m, override):
        initial = override / 100.0 if override is not None else segv / 100.0
        total = initial + S0v
        capped = jS < (m - 1)
        return torch.where(capped, torch.minimum(total, torch.ones_like(total)), total)
    # ---- stage 3: note geometry --------------------------------------------------
    def note_geometry(self, D):
        NJS, JD, height = D["NJS"], D["JD"], D["height"]
        jump_duration = JD / NJS
        offset_x = (D["line_index"].to(self.dtype) - 1.5) * 0.6
        ybase = torch.full_like(offset_x, 0.25)
        jumpOffsetY = torch.clamp(torch.as_tensor((height - 1.7999999523162842) * 0.5,
                                                  device=self.device, dtype=self.dtype), -0.2, 0.6)
        highest = torch.where(D["line_layer"] == 0, torch.tensor(0.85, device=self.device, dtype=self.dtype),
                              torch.where(D["line_layer"] == 1, torch.tensor(1.4, device=self.device, dtype=self.dtype),
                                          torch.tensor(1.9, device=self.device, dtype=self.dtype))) + jumpOffsetY
        num = JD / NJS * 0.5
        g = 2.0 * (highest - ybase) / (num * num)
        start_vv = g * jump_duration / 2.0
        jump_start = D["note_time"] - jump_duration / 2.0

        cd = D["cut_dir"]
        end_angle = torch.where(cd == 0, torch.tensor(-180.0, device=self.device, dtype=self.dtype),
                      torch.where(cd == 1, torch.tensor(0.0, device=self.device, dtype=self.dtype),
                      torch.where(cd == 2, torch.tensor(-90.0, device=self.device, dtype=self.dtype),
                      torch.where(cd == 3, torch.tensor(90.0, device=self.device, dtype=self.dtype),
                      torch.where(cd == 4, torch.tensor(-135.0, device=self.device, dtype=self.dtype),
                      torch.where(cd == 5, torch.tensor(135.0, device=self.device, dtype=self.dtype),
                      torch.where(cd == 6, torch.tensor(-45.0, device=self.device, dtype=self.dtype),
                      torch.where(cd == 7, torch.tensor(45.0, device=self.device, dtype=self.dtype),
                                  torch.zeros_like(D["note_time"])))))))))
        end_rot = q_from_euler(torch.zeros_like(end_angle), torch.zeros_like(end_angle), end_angle)
        euler = q_to_euler(end_rot)
        index = torch.remainder(torch.abs(torch.round(D["note_time"] * 10 + offset_x * 2 + ybase * 2)).to(torch.int64), 10)
        euler = euler + RANDOM_ROTATIONS.to(device=self.device).to(self.dtype)[index] * 20.0
        middle_rot = q_from_euler(euler[..., 0], euler[..., 1], euler[..., 2])

        floor_end = torch.stack([
            offset_x,
            torch.full_like(offset_x, 0.25),
            torch.full_like(offset_x, 0.65 + JD / 2 + 0.25),
        ], dim=-1)
        jump_end = torch.stack([
            offset_x,
            torch.full_like(offset_x, 0.25),
            torch.full_like(offset_x, 0.65 - JD / 2 + 0.25),
        ], dim=-1)
        return {
            "jump_duration": torch.as_tensor(jump_duration, device=self.device, dtype=self.dtype),
            "jump_start": jump_start, "g": g, "start_vv": start_vv,
            "middle_rot": middle_rot, "end_rot": end_rot,
            "floor_end": floor_end, "jump_end": jump_end,
        }

    def note_orientations(self, D, nc, e_idx, n_idx, r_e):
        t = D["t"]
        F = t.shape[0]
        E = e_idx.shape[0]
        dev = self.device
        f_c = r_e - 1
        f_c_c = torch.clamp(f_c, 0, F - 1)
        pct_c = (t[f_c_c] - nc["jump_start"][n_idx]) / nc["jump_duration"]

        rel_c = t[f_c_c] - nc["jump_start"][n_idx]
        local_y = nc["floor_end"][n_idx, 1] + nc["start_vv"][n_idx] * rel_c - nc["g"][n_idx] * rel_c ** 2 * 0.5
        headZ = D["head"][f_c_c, 2]
        z_c = _lerp_unclamped(nc["floor_end"][n_idx, 2] + headZ * torch.minimum(torch.ones_like(pct_c), pct_c * 2),
                              nc["jump_end"][n_idx, 2] + headZ, pct_c)
        num2 = (pct_c - 0.75) / 0.25
        shift = torch.where(pct_c >= 0.75, _lerp_unclamped(torch.zeros_like(num2), torch.full_like(num2, 500.0), num2 ** 3),
                            torch.zeros_like(num2))
        pos = torch.stack([nc["floor_end"][n_idx, 0], local_y, z_c - shift], dim=-1)
        valid_pos = (f_c >= 0) & (pct_c >= 0)
        pos = torch.where(valid_pos[:, None], pos, torch.zeros_like(pos))

        # rotation recurrence (fixed K iterations -> CUDA-graph friendly)
        s_n = torch.searchsorted(t, nc["jump_start"][n_idx], side="left")
        half_n = torch.searchsorted(t, nc["jump_start"][n_idx] + nc["jump_duration"] * 0.5, side="left")
        end_n = torch.minimum(half_n, r_e)
        L = torch.clamp(end_n - s_n, min=0)
        K = int(L.max().item()) if E else 0
        rot = torch.zeros((E, 4), device=dev, dtype=self.dtype)
        rot[:, 3] = 1.0
        up0 = torch.zeros((E, 3), device=dev, dtype=self.dtype)
        up0[:, 1] = 1.0
        start_rot = rot.clone()
        zvec = torch.zeros((E, 3), device=dev, dtype=self.dtype)
        zvec[:, 2] = 1.0
        for k in range(K):
            f = s_n + k
            fc = torch.clamp(f, 0, F - 1)
            valid = k < L
            pct = (t[fc] - nc["jump_start"][n_idx]) / nc["jump_duration"]
            rel = t[fc] - nc["jump_start"][n_idx]
            local_y = nc["floor_end"][n_idx, 1] + nc["start_vv"][n_idx] * rel - nc["g"][n_idx] * rel ** 2 * 0.5
            headZ = D["head"][fc, 2]
            z = _lerp_unclamped(nc["floor_end"][n_idx, 2] + headZ * torch.minimum(torch.ones_like(pct), pct * 2),
                                nc["jump_end"][n_idx, 2] + headZ, pct)
            lp = torch.stack([nc["floor_end"][n_idx, 0], local_y, z], dim=-1)
            hpx = D["head"][fc, 0]
            hpy = _lerp(D["head"][fc, 1], local_y, 0.8)
            hp = torch.stack([hpx, hpy, headZ], dim=-1)
            direction = v_norm(lp - hp)
            a = torch.where((pct >= 0.125)[:, None],
                            q_slerp(nc["middle_rot"][n_idx], nc["end_rot"][n_idx], torch.sin((pct - 0.125) * 2 * PI)),
                            q_slerp(start_rot, nc["middle_rot"][n_idx], torch.sin(pct * 4 * PI)))
            rup = q_rot(up0, rot)
            b = look_rotation(direction, rup)
            new_rot = q_lerp(a, b, pct * 2)
            rot = torch.where(valid[:, None], new_rot, rot)
        fwd = q_rot(zvec, rot)
        return pos, rot, fwd

    # ---- forward ----------------------------------------------------------------
    def forward(self, D):
        t = D["t"]
        F = t.shape[0]
        e_idx = D["e_idx"]
        n_idx = D["n_idx"]
        r_e = D["r_e"]
        hand = D["hand"]
        E = e_idx.shape[0]

        nl = self.saber_arrays(D, D["hiltL"], D["tipL"])
        nr = self.saber_arrays(D, D["hiltR"], D["tipR"])
        normals = torch.stack([nl[0], nr[0]])
        segs = torch.stack([nl[1], nr[1]])
        S0L, jSL = self.swing_precompute(nl[0], nl[1], t, D["J"])
        S0R, jSR = self.swing_precompute(nr[0], nr[1], t, D["J"])
        S0 = torch.stack([S0L, S0R])
        jS = torch.stack([jSL, jSR])

        nc = self.note_geometry(D)
        pos, rot, fwd = self.note_orientations(D, nc, e_idx, n_idx, r_e)

        cpn = normals[hand, r_e]
        cut_point = D["cut_point"]

        # accuracy
        denom = v_dot(cpn, cpn)
        dist = torch.where(denom == 0, torch.zeros_like(denom),
                           torch.abs(v_dot(cpn, cut_point - pos)))
        acc_pct = torch.where(dist > 0.3, torch.zeros_like(dist), 1.0 - dist / 0.3)
        acc_float = acc_pct * 15.0
        acc = torch.round(acc_float).to(torch.int64)

        # registration before-rating
        S0m = S0[hand, r_e]
        jSm = jS[hand, r_e]
        segm = segs[hand, r_e]
        before0 = self.swing_batch(S0m, jSm, segm, r_e, None)

        # per-event state
        before = before0
        after = torch.zeros(E, device=self.device, dtype=self.dtype)
        cut_time = t[r_e].clone()
        has_cut = torch.zeros(E, dtype=torch.bool, device=self.device)
        finished = torch.zeros(E, dtype=torch.bool, device=self.device)

        # registration order per frame
        reg_by_frame = {}
        for i in range(E):
            reg_by_frame.setdefault(int(r_e[i]), []).append(i)
        f_start = int(r_e.min()) if E else F
        active = []
        active_ids = None
        active_dirty = False
        combo = 0
        score = 0.0
        rec = []
        g_before = []
        g_after = []
        g_acc = []
        g_mult = []

        def multiplier(meter):
            if meter == 0:
                return 1
            if meter < 5:
                return 2
            if meter < 13:
                return 4
            return 8

        for f in range(f_start, F):
            if f in reg_by_frame:
                active.extend(reg_by_frame[f])
                active_dirty = True
            if not active:
                continue
            if f == 0:
                continue
            if active_dirty or active_ids is None or active_ids.shape[0] != len(active):
                active_ids = torch.as_tensor(active, dtype=torch.int64, device=self.device)
                active_dirty = False
            Ea = active_ids.shape[0]
            if Ea == 0:
                continue

            # ---- batched provisional update at frame f for all active events ----
            h = hand[active_ids]
            delta = t[f] - cut_time[active_ids]
            fin_delta = delta > 0.4

            tipL = D["tipL"]; tipR = D["tipR"]; hiltL = D["hiltL"]; hiltR = D["hiltR"]
            tip_p = torch.where((h == 0)[:, None], tipL[f - 1], tipR[f - 1])
            tip_c = torch.where((h == 0)[:, None], tipL[f], tipR[f])
            hilt_p = torch.where((h == 0)[:, None], hiltL[f - 1], hiltR[f - 1])
            hilt_c = torch.where((h == 0)[:, None], hiltL[f], hiltR[f])

            cpn_e = cpn[active_ids]
            notePos = pos[active_ids]
            fwd_e = fwd[active_ids]
            np_ = v_cross(cpn_e, fwd_e)
            s_prev = torch.sign(v_dot(np_, tip_p - notePos))
            s_curr = torch.sign(v_dot(np_, tip_c - notePos))
            crossing = (s_curr != s_prev) & (~has_cut[active_ids]) & (~fin_delta)

            denom_r = v_dot(np_, tip_c - tip_p)
            safe_d = torch.where(denom_r == 0, torch.ones_like(denom_r), denom_r)
            dist_r = torch.where(denom_r == 0, torch.zeros_like(denom_r),
                                 v_dot(np_, notePos - tip_p) / safe_d)
            cut_tip = torch.where((s_prev == 0)[:, None], tip_p,
                                  tip_p + dist_r[:, None] * (tip_c - tip_p))
            cut_hilt = (hilt_p + hilt_c) / 2.0
            be_err = v_angle(cut_tip - cut_hilt, tip_p - hilt_p)
            ae_err = v_angle(cut_tip - cut_hilt, tip_c - hilt_c)

            S0f = S0[h, f]
            jSf = jS[h, f]
            segf = segs[h, f]
            swing_cross = self.swing_batch(S0f, jSf, segf, f,
                                           be_err)

            ang = v_angle(cpn_e, normals[h, f])
            fin_ang = ang >= 90.0
            w = torch.where(ang < 75.0, torch.ones_like(ang), (90.0 - ang) / 15.0)
            after_inc = segf * w / 60.0

            after_path = (~fin_delta) & (~crossing)
            nb = torch.where(crossing, swing_cross, before[active_ids])
            na = torch.where(crossing, ae_err / 60.0, after[active_ids])
            na = torch.where(after_path & fin_ang, after[active_ids], na)
            na_inc = after[active_ids] + after_inc
            na = torch.where(after_path & (~fin_ang), na_inc, na)
            over = after_path & (~fin_ang) & (na_inc > 1.0)
            na = torch.where(over, torch.ones_like(na), na)
            nfin = finished[active_ids] | fin_delta | (after_path & fin_ang) | over
            nct = torch.where(crossing, t[f], cut_time[active_ids])
            nhc = has_cut[active_ids] | crossing

            # ---- score loop with exact reference iteration semantics ----
            # the provisional flags are keyed by ORIGINAL active order; the
            # active list mutates under pop(), so track events by identity.
            if not bool(torch.any(nfin).item()):
                continue
            orig_pos = {e: p for p, e in enumerate(active)}
            nfin_np = nfin.detach().cpu().numpy()
            visited = np.zeros(Ea, dtype=bool)
            scored = []  # (orig_position, event)
            i = 0
            while i < len(active):
                e = active[i]
                p = orig_pos[e]
                visited[p] = True
                if nfin_np[p]:
                    scored.append((p, e))
                    active.pop(i)
                    i += 1  # iterator advances -> skips next element
                else:
                    i += 1
            if scored:
                pos_s = torch.as_tensor([p for p, _ in scored], dtype=torch.int64, device=self.device)
                active_dirty = True
                nb_s = nb[pos_s]
                na_s = na[pos_s]
                acc_s = acc[active_ids[pos_s]]
                accf_s = acc_float[active_ids[pos_s]]
                for j_, (p, e) in enumerate(scored):
                    mult = multiplier(combo)
                    s = int(round(nb_s[j_].item() * 70)) + int(round(na_s[j_].item() * 30)) + int(acc_s[j_].item())
                    score += s * mult
                    combo += 1
                    g_before.append(nb_s[j_])
                    g_after.append(na_s[j_])
                    g_acc.append(accf_s[j_])
                    g_mult.append(mult)
                    rec.append({
                        "n_idx": int(n_idx[e].item()),
                        "before": nb_s[j_].item(), "after": na_s[j_].item(),
                        "acc": int(acc_s[j_].item()),
                        "breakdown": [int(round(nb_s[j_].item() * 70)), int(round(na_s[j_].item() * 30)), int(acc_s[j_].item())],
                        "mult": mult, "path": "update",
                    })

            # ---- persist state for visited events (skipped keep prior) ----
            vi = torch.nonzero(torch.as_tensor(visited, device=self.device)).flatten()
            if vi.numel():
                e_ids = active_ids[vi]
                before = before.index_copy(0, e_ids, nb[vi])
                after = after.index_copy(0, e_ids, na[vi])
                cut_time = cut_time.index_copy(0, e_ids, nct[vi])
                has_cut = has_cut.index_copy(0, e_ids, nhc[vi])
                finished = finished.index_copy(0, e_ids, nfin[vi])

        # finish(): increment-then-score, same iteration semantics
        i = 0
        while i < len(active):
            e = active[i]
            combo += 1
            mult = multiplier(combo)
            s = int(round(before[e].item() * 70)) + int(round(after[e].item() * 30)) + int(acc[e].item())
            score += s * mult
            g_before.append(before[e])
            g_after.append(after[e])
            g_acc.append(acc_float[e])
            g_mult.append(mult)
            rec.append({
                "n_idx": int(n_idx[e].item()),
                "before": before[e].item(), "after": after[e].item(),
                "acc": int(acc[e].item()),
                "breakdown": [int(round(before[e].item() * 70)), int(round(after[e].item() * 30)), int(acc[e].item())],
                "mult": mult, "path": "finish",
            })
            active.pop(i)
            i += 1

        # ---- differentiable smooth loss tensors (per-scored-event, scoring order) ----
        if g_before:
            fin_before = torch.stack(g_before)
            fin_after = torch.stack(g_after)
            fin_acc = torch.stack(g_acc)
            mult_t = torch.as_tensor(g_mult, dtype=self.dtype, device=self.device)
            smooth_total = torch.sum((fin_before * 70.0 + fin_after * 30.0 + fin_acc) * mult_t)
        else:
            fin_before = torch.zeros(0, dtype=self.dtype, device=self.device)
            fin_after = torch.zeros(0, dtype=self.dtype, device=self.device)
            fin_acc = torch.zeros(0, dtype=self.dtype, device=self.device)
            mult_t = torch.zeros(0, dtype=self.dtype, device=self.device)
            smooth_total = torch.zeros((), dtype=self.dtype, device=self.device)

        result = SimpleNamespace(
            before=fin_before,
            after=fin_after,
            acc_float=fin_acc,
            acc=torch.round(fin_acc).to(torch.int64),
            mult=mult_t,
            total=score,
            smooth_total=smooth_total,
            rec=rec,
        )
        return result


class SimpleNamespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run_map(map_dir, replay_path, device=None):
    D = preprocess(map_dir, replay_path)
    scorer = SaberScorer(device)
    TD = to_tensor(D, device)
    res = scorer(TD)
    return D, res
