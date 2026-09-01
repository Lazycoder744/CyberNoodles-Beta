"""GeneratorSmaller3 port + upstream custom loss (bsgen).

Faithful torch port of third_party/bs-replay-generator/pytorch_walls_onnx_fix3.py
(GeneratorSmaller3, KerasStyle{Self}MHA, custom_loss_with_angle_2_torch).
"""
import math

import torch
from torch import nn

FRAME_COUNT_INPUT = 50
FRAME_COUNT_OUTPUT = 50
FRAME_LENGTH = 22
EXTRA_STATS = 6

NOTE_COUNT = 50
NOTE_LENGTH = 31

WALL_COUNT = 50
WALL_LENGTH = 6

NUM_TARGET_CHANNELS = FRAME_LENGTH - 1  # 21

EPS = 1e-6
_EPS = 1e-6

# Per-channel std of the target trajectory y across the training windows.
# The raw flow target u = y - x0 is dominated by x0 ~ N(0,1) (data std ~0.05-0.85),
# so the model collapses to a near-constant velocity field (mode collapse).
# Normalizing y by its per-channel std makes signal and noise comparable;
# the flow model then learns a well-conditioned velocity field in unit-variance space.
# Computed from /mnt/games/bs-ai/windows; floored at 0.05 for near-constant channels.
FLOW_SIGMA = torch.tensor([
    0.05, 0.063, 0.1124, 0.2574, 0.1095, 0.05, 0.05, 0.8437, 0.1851, 0.3466,
    0.3018, 0.5129, 0.4382, 0.3954, 0.5934, 0.1915, 0.3329, 0.3021, 0.5084,
    0.4539, 0.3887,
])


def set_seed(seed: int = 72583):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class KerasStyleMHA(nn.Module):
    """MultiHeadAttention(query=..., key=..., value=...) with Keras output_shape."""

    def __init__(self, q_dim, k_dim, v_dim, num_heads=32, key_dim=128, val_dim=128, out_dim=256):
        super().__init__()
        embed_dim = num_heads * key_dim
        self.q_proj = nn.Linear(q_dim, embed_dim)
        self.k_proj = nn.Linear(k_dim, key_dim)
        self.v_proj = nn.Linear(v_dim, val_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            kdim=key_dim, vdim=val_dim, batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, query_l, k_layer, v_layer):
        q = self.q_proj(query_l)
        k = self.k_proj(k_layer)
        v = self.v_proj(v_layer)
        attn_out, _ = self.mha(query=q, key=k, value=v, need_weights=False)
        return self.out_proj(attn_out)


class KerasStyleSelfMHA(nn.Module):
    """Self-attention with Keras-like signature: MHA(H, key_dim=D, output=O)."""

    def __init__(self, in_dim, num_heads, key_dim, out_dim):
        super().__init__()
        embed_dim = num_heads * key_dim
        self.in_proj = nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        h = self.in_proj(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        return self.out_proj(h)


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        shape = x.shape[:-1]
        y = self.module(x.contiguous().view(-1, x.size(-1)))
        return y.view(*shape, y.size(-1))


class GeneratorSmaller3(nn.Module):
    def __init__(self, notes_heads=16, notes_key_dim=128, dec_heads=16, dec_key_dim=128):
        super().__init__()
        self.frames_mha1 = KerasStyleSelfMHA(in_dim=FRAME_LENGTH + EXTRA_STATS, num_heads=4, key_dim=32, out_dim=8)
        self.frames_mha1_proj = nn.Linear(FRAME_COUNT_INPUT * 8, 8)

        self.notes_proj = TimeDistributed(nn.Linear(NOTE_LENGTH, 128))
        self.notes_proj_relu = nn.ReLU()
        self.notes_lstm_fwd = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.notes_mha1 = KerasStyleSelfMHA(in_dim=128, num_heads=notes_heads, key_dim=notes_key_dim, out_dim=128)
        self.notes_mha2 = KerasStyleSelfMHA(in_dim=128, num_heads=notes_heads, key_dim=notes_key_dim, out_dim=128)

        self.walls_proj = TimeDistributed(nn.Linear(WALL_LENGTH, 16))
        self.walls_proj_relu = nn.ReLU()
        self.walls_mha1 = KerasStyleSelfMHA(in_dim=16, num_heads=4, key_dim=16, out_dim=16)

        self.cross_notes = KerasStyleMHA(q_dim=1, k_dim=141, v_dim=141, num_heads=8, key_dim=16, val_dim=256, out_dim=256)
        self.cross_walls = KerasStyleMHA(q_dim=1, k_dim=29, v_dim=29, num_heads=2, key_dim=8, val_dim=8, out_dim=16)

        dec_in = 1 + EXTRA_STATS + 256 + 16 + 8
        self.dec_lstm1 = KerasStyleSelfMHA(in_dim=dec_in, num_heads=dec_heads, key_dim=dec_key_dim, out_dim=128)
        self.dec_lstm2 = KerasStyleSelfMHA(in_dim=128, num_heads=dec_heads, key_dim=dec_key_dim, out_dim=128)
        self.out = TimeDistributed(nn.Linear(128, NUM_TARGET_CHANNELS))

    def forward(self, frames_in, frame_times, extra_stats, notes_in, walls_in, note_time_diffs=None):
        B, T_in, _ = frames_in.shape
        T_out = frame_times.shape[1]

        extra_rep_in = extra_stats.unsqueeze(1).expand(-1, T_in, -1)
        extra_rep_out = extra_stats.unsqueeze(1).expand(-1, T_out, -1)

        x = torch.cat([frames_in, extra_rep_in], dim=-1)
        x = self.frames_mha1(x)
        x = x.reshape(B, -1)
        x = self.frames_mha1_proj(x)
        x_n = x.unsqueeze(1).expand(-1, NOTE_COUNT, -1)
        x_w = x.unsqueeze(1).expand(-1, WALL_COUNT, -1)
        x_f = x.unsqueeze(1).expand(-1, FRAME_COUNT_OUTPUT, -1)

        n = self.notes_proj(notes_in)
        n = self.notes_proj_relu(n)
        n, _ = self.notes_lstm_fwd(n)
        n = self.notes_mha1(n)
        n = self.notes_mha2(n)
        n_params = notes_in[:, :, :5]
        n = torch.cat([n, n_params, x_n], dim=-1)

        w = self.walls_proj(walls_in)
        w = self.walls_proj_relu(w)
        w = self.walls_mha1(w)
        w_params = walls_in[:, :, :5]
        w = torch.cat([w, w_params, x_w], dim=-1)

        notes_attn = self.cross_notes(frame_times, n, n)
        walls_attn = self.cross_walls(frame_times, w, w)

        dec_in = torch.cat([frame_times, extra_rep_out, notes_attn, walls_attn, x_f], dim=-1)
        z = self.dec_lstm1(dec_in)
        z = self.dec_lstm2(z)
        return self.out(z)


# ---------------------
# Upstream custom loss (weighted positions / tips / axes + quat-norm penalty)
# ---------------------

_BASE_MASK_50 = torch.tensor([
    16.9, 13.0, 10.0, 8.0, 6.9, 4.4, 3.1, 2.3, 1.5, 1.2,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
], dtype=torch.float32)


def _make_time_mask(frame_times):
    B, T, _ = frame_times.shape
    device = frame_times.device
    base = _BASE_MASK_50.to(device)
    if T < base.numel():
        base = base[:T]
    elif T > base.numel():
        base = torch.cat([base, torch.ones(T - base.numel(), device=device)], dim=0)
    return base.unsqueeze(0).expand(B, -1)


def _normalize_quaternion(q):
    norms = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(_EPS)
    return q / norms, norms


def _quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q.unbind(dim=-1)
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return R


def _calculate_saber_tip_position(hilt_pos, q, saber_length, rotation_length):
    R = _quaternion_to_rotation_matrix(q)
    v_tip = torch.tensor([0.0, 0.0, -saber_length], dtype=R.dtype, device=R.device)
    v_y = torch.tensor([0.0, rotation_length, 0.0], dtype=R.dtype, device=R.device)
    v_x = torch.tensor([rotation_length, 0.0, 0.0], dtype=R.dtype, device=R.device)
    tip = hilt_pos + (R @ v_tip).squeeze(-1)
    n_y = (R @ v_y).squeeze(-1)
    n_x = (R @ v_x).squeeze(-1)
    return tip, n_y, n_x


@torch.no_grad()
def _prep_note_weights(frame_times, note_times, note_time_diffs, k_tip, k_pos):
    ft = frame_times
    nd = torch.where(note_time_diffs == 0,
                     torch.tensor(6969.69, device=ft.device, dtype=ft.dtype),
                     note_time_diffs)
    nt = torch.where(note_times == 0,
                     torch.tensor(696969.69, device=note_times.device, dtype=note_times.dtype),
                     note_times)
    nt = nt - nd
    abs_diff = (ft - nt.unsqueeze(1)).abs()
    lowest = abs_diff.min(dim=-1).values
    base = _make_time_mask(frame_times)
    tip = base * torch.clamp(1.0 - k_tip * lowest, min=0.3).pow(2)
    pos = base * torch.clamp(1.0 - k_pos * lowest, min=0.3).pow(2)
    return tip, pos


def custom_loss_with_angle_2_torch(y_true, y_pred, frame_times, note_times_left, note_times_right,
                                   note_time_diffs, saber_length=1.0, return_components=False,
                                   smooth_w=0.0):
    assert y_true.shape == y_pred.shape
    with torch.no_grad():
        base_mask = _make_time_mask(frame_times)
        tip_L, pos_L = _prep_note_weights(frame_times, note_times_left, note_time_diffs, k_tip=16.6, k_pos=6.9)
        tip_R, pos_R = _prep_note_weights(frame_times, note_times_right, note_time_diffs, k_tip=16.9, k_pos=6.9)

    headset_pos_true, headset_pos_pred = y_true[..., 0:3], y_pred[..., 0:3]
    left_hilt_true, left_hilt_pred = y_true[..., 7:10], y_pred[..., 7:10]
    right_hilt_true, right_hilt_pred = y_true[..., 14:17], y_pred[..., 14:17]

    headset_q_true, _ = _normalize_quaternion(y_true[..., 3:7])
    headset_q_pred, hn = _normalize_quaternion(y_pred[..., 3:7])
    left_q_true, _ = _normalize_quaternion(y_true[..., 10:14])
    left_q_pred, ln = _normalize_quaternion(y_pred[..., 10:14])
    right_q_true, _ = _normalize_quaternion(y_true[..., 17:21])
    right_q_pred, rn = _normalize_quaternion(y_pred[..., 17:21])

    norm_loss = (1.0 - hn).abs().mean() * 0.2 + (1.0 - ln).abs().mean() + (1.0 - rn).abs().mean()

    headset_tip_true, headset_y_true, headset_x_true = _calculate_saber_tip_position(
        headset_pos_true, headset_q_true, saber_length, saber_length)
    headset_tip_pred, headset_y_pred, headset_x_pred = _calculate_saber_tip_position(
        headset_pos_pred, headset_q_pred, saber_length, saber_length)
    left_tip_true, left_y_true, left_x_true = _calculate_saber_tip_position(
        left_hilt_true, left_q_true, saber_length, saber_length)
    left_tip_pred, left_y_pred, left_x_pred = _calculate_saber_tip_position(
        left_hilt_pred, left_q_pred, saber_length, saber_length)
    right_tip_true, right_y_true, right_x_true = _calculate_saber_tip_position(
        right_hilt_true, right_q_true, saber_length, saber_length)
    right_tip_pred, right_y_pred, right_x_pred = _calculate_saber_tip_position(
        right_hilt_pred, right_q_pred, saber_length, saber_length)

    def sqdist_per_frame(a, b):
        return ((a - b) ** 2).sum(dim=-1)

    pos_loss = (
        (sqdist_per_frame(headset_pos_true, headset_pos_pred) * base_mask).mean() * 0.2
        + (sqdist_per_frame(left_hilt_true, left_hilt_pred) * pos_L).mean()
        + (sqdist_per_frame(right_hilt_true, right_hilt_pred) * pos_R).mean()
    )
    tip_pos_loss = (
        (sqdist_per_frame(headset_tip_true, headset_tip_pred) * base_mask).mean() * 0.15
        + (sqdist_per_frame(left_tip_true, left_tip_pred) * tip_L).mean()
        + (sqdist_per_frame(right_tip_true, right_tip_pred) * tip_R).mean()
    )
    angle_diff_y = (
        (sqdist_per_frame(headset_y_true, headset_y_pred) * base_mask).mean() * 0.15
        + (sqdist_per_frame(left_y_true, left_y_pred) * tip_L).mean()
        + (sqdist_per_frame(right_y_true, right_y_pred) * tip_R).mean()
    )
    angle_diff_x = (
        (sqdist_per_frame(headset_x_true, headset_x_pred) * base_mask).mean() * 0.15
        + (sqdist_per_frame(left_x_true, left_x_pred) * tip_L).mean()
        + (sqdist_per_frame(right_x_true, right_x_pred) * tip_R).mean()
    )

    loss = (norm_loss ** 2) * 0.05 + (pos_loss * 2.0) + tip_pos_loss + (angle_diff_y + angle_diff_x) * 0.15
    if smooth_w and smooth_w > 0.0:
        smooth, scomp = velocity_smoothness_loss(y_pred, frame_times, return_components=True)
        loss = loss + smooth_w * smooth
    if return_components:
        comp = {
            "norm": norm_loss.item(),
            "pos": pos_loss.item(),
            "tip": tip_pos_loss.item(),
            "angle_y": angle_diff_y.item(),
            "angle_x": angle_diff_x.item(),
        }
        if smooth_w and smooth_w > 0.0:
            comp.update({"sm_head": scomp["head"], "sm_left": scomp["left"], "sm_right": scomp["right"]})
        return loss, comp
    return loss


def note_times_from_x2(notes_in):
    """Left/right note hit times from x2 encoding (upstream derivation)."""
    note_times_left = notes_in[..., 0] * torch.max(notes_in[..., 8:18], dim=-1).values
    note_times_right = notes_in[..., 0] * torch.max(notes_in[..., 21:31], dim=-1).values
    return note_times_left, note_times_right


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------
# Flow-matching policy (stochastic Transformer) for BC + RL + flow selection
# ---------------------

def _t_embedding(t, dim):
    """Sinusoidal embedding of flow time t in [0,1]. t: (B,) -> (B, dim)."""
    freqs = torch.exp(torch.arange(0, dim, 2, device=t.device).float()
                      * (-math.log(10000.0) / max(1, dim - 1)))
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowGeneratorSmaller3(nn.Module):
    """Stochastic flow-matching policy on the same Transformer encoder as GeneratorSmaller3.

    Encodes the same context (past frames, upcoming notes/walls) but instead of
    predicting the trajectory directly, predicts a velocity field v(x_t, t, ctx)
    for the conditional flow (1-t)*x0 + t*y. At rollout the ODE is integrated
    from noise x0 ~ N(0, I); fresh noise => different replay. Flow selection at
    train/eval samples K candidates and keeps the best (or softmax-weighted).
    """

    def __init__(self, t_dim=16, notes_heads=16, notes_key_dim=128, dec_heads=16, dec_key_dim=128):
        super().__init__()
        self.t_dim = t_dim
        # ---- encoder: identical to GeneratorSmaller3 ----
        self.frames_mha1 = KerasStyleSelfMHA(in_dim=FRAME_LENGTH + EXTRA_STATS, num_heads=4, key_dim=32, out_dim=8)
        self.frames_mha1_proj = nn.Linear(FRAME_COUNT_INPUT * 8, 8)

        self.notes_proj = TimeDistributed(nn.Linear(NOTE_LENGTH, 128))
        self.notes_proj_relu = nn.ReLU()
        self.notes_lstm_fwd = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.notes_mha1 = KerasStyleSelfMHA(in_dim=128, num_heads=notes_heads, key_dim=notes_key_dim, out_dim=128)
        self.notes_mha2 = KerasStyleSelfMHA(in_dim=128, num_heads=notes_heads, key_dim=notes_key_dim, out_dim=128)

        self.walls_proj = TimeDistributed(nn.Linear(WALL_LENGTH, 16))
        self.walls_proj_relu = nn.ReLU()
        self.walls_mha1 = KerasStyleSelfMHA(in_dim=16, num_heads=4, key_dim=16, out_dim=16)

        self.cross_notes = KerasStyleMHA(q_dim=1, k_dim=141, v_dim=141, num_heads=8, key_dim=16, val_dim=256, out_dim=256)
        self.cross_walls = KerasStyleMHA(q_dim=1, k_dim=29, v_dim=29, num_heads=2, key_dim=8, val_dim=8, out_dim=16)

        # ---- flow decoder: x_t + t_emb + context -> velocity (B, 50, 21) ----
        dec_in = NUM_TARGET_CHANNELS + t_dim + EXTRA_STATS + 256 + 16 + 8
        self.dec_lstm1 = KerasStyleSelfMHA(in_dim=dec_in, num_heads=dec_heads, key_dim=dec_key_dim, out_dim=128)
        self.dec_lstm2 = KerasStyleSelfMHA(in_dim=128, num_heads=dec_heads, key_dim=dec_key_dim, out_dim=128)
        self.out = TimeDistributed(nn.Linear(128, NUM_TARGET_CHANNELS))
        self.t_proj = nn.Linear(t_dim, t_dim)

    # ---- context encoding (shared, no grad bottleneck) ----
    def encode_context(self, frames_in, frame_times, extra_stats, notes_in, walls_in):
        B, T_in, _ = frames_in.shape
        T_out = frame_times.shape[1]

        extra_rep_in = extra_stats.unsqueeze(1).expand(-1, T_in, -1)
        extra_rep_out = extra_stats.unsqueeze(1).expand(-1, T_out, -1)

        x = torch.cat([frames_in, extra_rep_in], dim=-1)
        x = self.frames_mha1(x)
        x = x.reshape(B, -1)
        x = self.frames_mha1_proj(x)
        x_n = x.unsqueeze(1).expand(-1, NOTE_COUNT, -1)
        x_w = x.unsqueeze(1).expand(-1, WALL_COUNT, -1)
        x_f = x.unsqueeze(1).expand(-1, FRAME_COUNT_OUTPUT, -1)

        n = self.notes_proj(notes_in)
        n = self.notes_proj_relu(n)
        n, _ = self.notes_lstm_fwd(n)
        n = self.notes_mha1(n)
        n = self.notes_mha2(n)
        n_params = notes_in[:, :, :5]
        n = torch.cat([n, n_params, x_n], dim=-1)

        w = self.walls_proj(walls_in)
        w = self.walls_proj_relu(w)
        w = self.walls_mha1(w)
        w_params = walls_in[:, :, :5]
        w = torch.cat([w, w_params, x_w], dim=-1)

        notes_attn = self.cross_notes(frame_times, n, n)
        walls_attn = self.cross_walls(frame_times, w, w)
        return {
            "frame_times": frame_times,
            "extra_rep_out": extra_rep_out,
            "notes_attn": notes_attn,
            "walls_attn": walls_attn,
            "x_f": x_f,
        }

    # ---- velocity field ----
    def velocity(self, x_t, t, ctx):
        """v(x_t, t, ctx): x_t (B,50,21), t (B,), ctx from encode_context -> (B,50,21)."""
        t_emb = _t_embedding(t, self.t_dim)
        t_emb = self.t_proj(t_emb).unsqueeze(1).expand(-1, FRAME_COUNT_OUTPUT, -1)
        dec_in = torch.cat([x_t, t_emb, ctx["extra_rep_out"], ctx["notes_attn"],
                            ctx["walls_attn"], ctx["x_f"]], dim=-1)
        z = self.dec_lstm1(dec_in)
        z = self.dec_lstm2(z)
        return self.out(z)

    # ---- ODE sampler (differentiable midpoint Euler, data-matched noise) ----
    def sample(self, ctx, noise=None, n_steps=16, t_end=1.0):
        """Integrate dx/dt = v from t=0..t_end starting at x(0)=noise (B,50,21).

        Noise is drawn at the per-channel data scale (sigma) so the ODE travels
        a distance comparable to the trajectory magnitude -- the velocity field
        is learned in this same real space, so the integration is well-conditioned.
        Evaluates velocity at the midpoint time (i+0.5)*dt for 2nd-order accuracy.
        """
        B = ctx["frame_times"].shape[0]
        if noise is None:
            noise = torch.randn(B, FRAME_COUNT_OUTPUT, NUM_TARGET_CHANNELS,
                                device=ctx["frame_times"].device, dtype=ctx["frame_times"].dtype)
            sigma = FLOW_SIGMA.to(noise.device)
            noise = noise * sigma
        x = noise
        dt = t_end / n_steps
        for i in range(n_steps):
            t = torch.full((B,), (i + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.velocity(x, t, ctx)
            x = x + dt * v
        return x

    # ---- deterministic eval forward (single Euler step path, zero noise) ----
    def forward(self, frames_in, frame_times, extra_stats, notes_in, walls_in, note_time_diffs=None):
        ctx = self.encode_context(frames_in, frame_times, extra_stats, notes_in, walls_in)
        noise = torch.zeros_like(ctx["frame_times"].expand(-1, FRAME_COUNT_OUTPUT, NUM_TARGET_CHANNELS))
        return self.sample(ctx, noise=noise, n_steps=1, t_end=1.0)


def velocity_smoothness_loss(y_pred, frame_times, return_components=False, group_weights=None):
    """Penalize frame-to-frame jerk (second temporal difference) of predicted trajectory.

    The dominant artifact of under-trained generators is per-frame teleportation
    (spikes in velocity), which appears as large second differences d2[i] =
    y[i+2] - 2*y[i+1] + y[i]. This term drives those to zero. Groups:
      - head  (channels 0:7)  weight 1.0  (natural head motion must be smooth)
      - left  (channels 7:14) weight 0.02 (light - prevents teleport but allows swings)
      - right (channels 14:21) weight 0.02
    Pass group_weights to override (e.g. scale whole term during training).
    """
    d1 = y_pred[:, 1:, :] - y_pred[:, :-1, :]          # velocity (B, T-1, 21)
    d2 = d1[:, 1:, :] - d1[:, :-1, :]                  # jerk    (B, T-2, 21)
    sq = d2 ** 2
    head = sq[..., 0:7]
    left = sq[..., 7:14]
    right = sq[..., 14:21]
    hw = group_weights.get("head", 1.0) if group_weights else 1.0
    lw = group_weights.get("left", 0.02) if group_weights else 0.02
    rw = group_weights.get("right", 0.02) if group_weights else 0.02
    loss = head.mean() * hw + left.mean() * lw + right.mean() * rw
    if return_components:
        return loss, {
            "head": head.mean().item() * hw,
            "left": left.mean().item() * lw,
            "right": right.mean().item() * rw,
        }
    return loss


def flow_matching_loss(y_true, x0_noise, v_pred, frame_times, note_times_left, note_times_right,
                       note_time_diffs, return_components=False, v_max=6.0):
    """Conditional flow-matching loss in real space with data-matched noise.

    x0_noise ~ N(0, sigma^2) (per-channel data std), so the target velocity
    u = y - x0 has comparable signal and noise magnitude (no mode collapse).
    The model's velocity output v_pred lives in the same real space. Weighted
    per-channel-group by the same near-note masks used by
    custom_loss_with_angle_2_torch.
    """
    u = (y_true - x0_noise).clamp(-v_max, v_max)
    assert v_pred.shape == u.shape, (v_pred.shape, u.shape)
    with torch.no_grad():
        base_mask = _make_time_mask(frame_times)
        _, pos_L = _prep_note_weights(frame_times, note_times_left, note_time_diffs, k_tip=16.6, k_pos=6.9)
        _, pos_R = _prep_note_weights(frame_times, note_times_right, note_time_diffs, k_tip=16.9, k_pos=6.9)

    d2 = (v_pred - u) ** 2
    head = d2[..., 0:7] * base_mask[..., None] * 0.2
    left = d2[..., 7:14] * pos_L[..., None]
    right = d2[..., 14:21] * pos_R[..., None]
    pos_loss = head.mean() + left.mean() + right.mean()

    if return_components:
        return pos_loss, {
            "head": head.mean().item(),
            "left": left.mean().item(),
            "right": right.mean().item(),
        }
    return pos_loss
