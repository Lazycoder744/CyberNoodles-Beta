# Sample usage:
# python pytorch_walls_onnx_fix3.py --model smaller_2 --data_dir ReplaysDownloads/processed-replays-test --epochs 20 --batch_size 256 --export_path ./model_ --num_workers 1

"""
pytorch_walls_onnx_fix3.py

- TensorFlow / tf2onnx / Keras FREE.
- Faithful conversion of make_generator_model_smaller_2 and make_generator_model_smaller_3.
- Streaming IterableDataset with memmap for large datasets.
- Physics loss and data mapping align with the original notebook.

Fixes vs previous version:
- smaller_3: cross-attn now takes notes K/V dim = 261 and walls K/V dim = 69 (after param concat).
- smaller_3: removed pre-projection of walls to 32; MHA handles output_shape=32 directly.
- Loss: left/right proximity weights use sentinel 696969.69 for "no note", and tip weights 16.6 (left) / 16.9 (right).
- Added per-epoch step caps `--train_steps`, `--val_steps`.
"""

import os
import glob
import math
import argparse
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# ---------------------
# Constants to mirror the notebook
# ---------------------
FRAME_COUNT_INPUT = 50
FRAME_COUNT_OUTPUT = 50
FRAME_LENGTH = 22
EXTRA_STATS = 6

NOTE_COUNT = 50
NOTE_LENGTH = 31

WALL_COUNT = 50
WALL_LENGTH = 6

# Per-note hit target ("target as hits"), see v1/scripts/window.py.
#   col0 present, col1 saber, col2 hit_time, col3-5 cut_normal, col6 result
HIT_VEC = 7

NUM_TARGET_CHANNELS = FRAME_LENGTH - 1  # 21

EPS = 1e-6

def set_seed(seed: int = 72583):
    torch.manual_seed(seed)
    np.random.seed(seed)

def device_from_arg(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Data loader (mirrors notebook)
# ---------------------
def _shard_sort_key(p: str) -> int:
    # numeric shard order: lexicographic sort puts "x0s_10.npy" BEFORE
    # "x0s_2.npy", silently mis-joining tokens to the wrong windows once a
    # dataset has ten or more shards (tokens are joined by sorted index).
    try:
        return int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0])
    except (ValueError, IndexError):
        return 1 << 30


def list_npy_groups(data_dir: str) -> List[Tuple[str, str, str, str, str, str]]:
    xs0 = sorted(glob.glob(os.path.join(data_dir, "x0s_*.npy")), key=_shard_sort_key)
    xs1 = sorted(glob.glob(os.path.join(data_dir, "x1s_*.npy")), key=_shard_sort_key)
    xs2 = sorted(glob.glob(os.path.join(data_dir, "x2s_*.npy")), key=_shard_sort_key)
    xs3 = sorted(glob.glob(os.path.join(data_dir, "x3s_*.npy")), key=_shard_sort_key)
    ys  = sorted(glob.glob(os.path.join(data_dir, "ys_*.npy")), key=_shard_sort_key)
    y2s = sorted(glob.glob(os.path.join(data_dir, "y2s_*.npy")), key=_shard_sort_key)
    n = min(len(xs0), len(xs1), len(xs2), len(xs3), len(ys), len(y2s))
    return list(zip(xs0[:n], xs1[:n], xs2[:n], xs3[:n], ys[:n], y2s[:n]))

def estimate_total_examples(file_groups: List[Tuple[str, str, str, str, str]]) -> int:
    total = 0
    for (x0p, _, _, _, _) in file_groups:
        try:
            arr = np.load(x0p, mmap_mode="r")
            total += int(arr.shape[0])
        except Exception:
            pass
    return total

def _filter_mask_batch(frames_in: np.ndarray, y_t: np.ndarray) -> np.ndarray:
    """
    Approximation of the original three tf.data ds.filter(...) clauses that gate
    on reasonable continuity of headset position between inputs (x0) and targets (y).
    Keep a sample only if, for each axis (x,y,z), the max per-timestep difference < 0.25.
    frames_in: (B, 50, 22),  positions at channels 1,2,3  (per the notebook)
    y_t:       (B, 50, 21),  headset_pos at channels 0,1,2
    """
    # Max |x0 - y| per sample, per axis
    dx = np.max(np.abs(frames_in[:, :, 1] - y_t[:, :, 0]), axis=1)  # (B,)
    dy = np.max(np.abs(frames_in[:, :, 2] - y_t[:, :, 1]), axis=1)
    dz = np.max(np.abs(frames_in[:, :, 3] - y_t[:, :, 2]), axis=1)

    mask = (dx < 0.25) & (dy < 0.25) & (dz < 0.25)

    # Also guard against NaN/Inf rows
    mask &= np.isfinite(frames_in).all(axis=(1, 2))
    mask &= np.isfinite(y_t).all(axis=(1, 2))
    return mask

class WallsIterableDataset(IterableDataset):
    """
    Yields per-sample batches from memory-mapped npy groups, matching the notebook:
    - frames_in = x0[:, :50, :22]
    - frame_times = x1[:, :50][..., None]
    - extra_stats = x1[:, 50:]
    - notes_in = x2[:, :50, :37]
    - walls_in = x3[:, :50, :6]
    - note_time_diffs = x2[:, :, 4]
    """
    def __init__(self, file_groups, batch_size: int, repeat=True, shuffle_files=True, shuffle_within=True,
                 time_warp=(1.0, 1.0), x0_noise=0.0):
        super().__init__()
        self.file_groups = list(file_groups)
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle_files = shuffle_files
        self.shuffle_within = shuffle_within
        # Augmentations:
        # - time_warp (smin, smax): log-uniform tempo scale; re-times dt/frame/note/wall
        #   channels and scales the njs stat (matches BS modifier semantics). Fixes the
        #   speed-skill bias without new data.
        # - x0_noise: max sigma of zero-mean noise added to context frames each window
        #   (sigma ~ U(0, max)); makes the model robust to its own rollout drift
        #   (exposure bias) without needing contiguous free-running chains.
        self.time_warp = (float(time_warp[0]), float(time_warp[1]))
        self.x0_noise = float(x0_noise)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        nworkers = worker.num_workers if worker is not None else 1
        rng = np.random.default_rng(72583 + wid)
        while True:
            order = np.arange(len(self.file_groups))
            if self.shuffle_files:
                rng.shuffle(order)
            # multi-worker correctness: without this stride every worker
            # iterates the FULL group list and each epoch sees the data
            # num_workers times over (silently duplicated batches)
            order = order[wid::nworkers]
            for idx in order:
                x0p, x1p, x2p, x3p, yp, y2p = self.file_groups[idx]
                # memory-map the read-only arrays: fancy indexing (x0[sl, ...])
                # copies just the batch, so a worker's RSS stays ~MBs instead
                # of one eager ~0.5GB group load per shard.  x2 stays eager --
                # the note-smoothing below mutates it in place.
                x0 = np.load(x0p, mmap_mode="r")
                x1 = np.load(x1p, mmap_mode="r")
                x2 = np.load(x2p)
                x3 = np.load(x3p, mmap_mode="r")
                y  = np.load(yp, mmap_mode="r")
                y2 = np.load(y2p, mmap_mode="r")
                timings = x2[:, :, 4].astype(np.float32, copy=True)
                # print(x2[0, :10, 1])
                weights = np.maximum(0, 1 - 25 * np.abs(x2[:, :, [0]] - np.reshape(x2[:, :, [0]], (len(x2), 1, NOTE_COUNT)))) * (np.where(x2[:, :, [0]] == 0, [0], [1]) * np.reshape(np.where(x2[:, :, [4]] == 0, [0], [1]), (len(x2), 1, NOTE_COUNT)))
                # print(weights)


                x2[:, :, 1] = np.sum(np.reshape(x2[:, :, [1]], (len(x2), 1, NOTE_COUNT)) * weights, axis=2) / (np.sum(weights, axis=2) + EPS)
                x2[:, :, 2] = np.sum(np.reshape(x2[:, :, [2]], (len(x2), 1, NOTE_COUNT)) * weights, axis=2) / (np.sum(weights, axis=2) + EPS)
                # x2[:, :, 3] = np.sum(np.reshape(x2[:, :, [3]], (len(x2), 1, NOTE_COUNT)) * weights, axis=2) / (np.sum(weights, axis=2) + EPS)
                x2[:, :, 4] = np.sum(np.reshape(x2[:, :, [4]], (len(x2), 1, NOTE_COUNT)) * weights, axis=2) / (np.sum(weights, axis=2) + EPS)
                # print(x2[0, :10, 1])

                B = x0.shape[0]
                indices = np.arange(B)
                if self.shuffle_within:
                    rng.shuffle(indices)

                for s in range(0, B, self.batch_size):
                    sl = indices[s:s+self.batch_size]
                    frames_in = x0[sl, :FRAME_COUNT_INPUT, :FRAME_LENGTH].astype(np.float32, copy=False)
                    frame_times = x1[sl, :FRAME_COUNT_OUTPUT][..., np.newaxis].astype(np.float32, copy=False)
                    extra_stats = x1[sl, FRAME_COUNT_OUTPUT:].astype(np.float32, copy=False)

                    # --- NEW: load notes/walls, then apply the notebook's weights-based smoothing ---
                    notes_in = x2[sl, :NOTE_COUNT, :NOTE_LENGTH].astype(np.float32, copy=False)
                    walls_in = x3[sl, :WALL_COUNT, :WALL_LENGTH].astype(np.float32, copy=False)

                    # # NOTE: this modifies notes_in in-place to match the original weights logic
                    # _smooth_notes_inplace(notes_in)

                    # After smoothing, take timing diffs from the (updated) channel 4
                    note_time_diffs = timings[sl, :].astype(np.float32, copy=False)

                    # Targets
                    # y_t = x0[sl, :FRAME_COUNT_OUTPUT, 1:].astype(np.float32, copy=False)
                    y_t = y[sl, :FRAME_COUNT_OUTPUT, :NUM_TARGET_CHANNELS].astype(np.float32, copy=False)
                    y2_t = y2[sl, :NOTE_COUNT, :HIT_VEC].astype(np.float32, copy=False)

                    # # --- NEW: apply the original ds.filter gating (per-sample mask) ---
                    keep = _filter_mask_batch(frames_in, y_t)
                    if not np.any(keep):
                        continue  # skip this slice entirely if nothing passes

                    frames_in       = frames_in[keep]
                    frame_times     = frame_times[keep]
                    extra_stats     = extra_stats[keep]
                    notes_in        = notes_in[keep]
                    walls_in        = walls_in[keep]
                    note_time_diffs = note_time_diffs[keep]
                    y_t             = y_t[keep]
                    y2_t            = y2_t[keep]

                    # --- augmentations (after filtering; boolean indexing already copied) ---
                    wlo, whi = self.time_warp
                    if wlo != 1.0 or whi != 1.0:
                        s = np.exp(rng.uniform(np.log(wlo), np.log(whi), size=len(frames_in))).astype(np.float32)
                        frames_in[:, :, 0] *= s[:, None]      # frame dt
                        frame_times = frame_times * s[:, None, None]  # output grid
                        extra_stats[:, 0] *= s                # njs stat
                        notes_in[:, :, 0] *= s[:, None]       # note times
                        walls_in[:, :, 0] *= s[:, None]       # wall times
                        y2_t[:, :, 2] *= s[:, None]           # hit_time scales with tempo
                    if self.x0_noise > 0.0:
                        sigma = float(rng.uniform(0.0, self.x0_noise))
                        if sigma > 0.0:
                            # jitter spatial state only; keep the dt column exact so the
                            # model's internal velocity/accel features stay consistent
                            frames_in[:, 1:] += rng.normal(
                                0.0, sigma, frames_in[:, 1:].shape).astype(np.float32)


                    inputs = (
                        torch.from_numpy(frames_in),
                        torch.from_numpy(frame_times),
                        torch.from_numpy(extra_stats),
                        torch.from_numpy(notes_in),
                        torch.from_numpy(walls_in),
                        torch.from_numpy(note_time_diffs),
                    )
                    target = torch.from_numpy(y_t)
                    hit_target = torch.from_numpy(y2_t)
                    # torch.set_printoptions(profile="full")
                    # for input in inputs:
                    #     print(input.shape)
                    #     print(input)
                    # print("target:")
                    # print(target[0])
                    # break
                    yield inputs, target, hit_target

            if not self.repeat:
                break

# ---------------------
# Attention blocks that emulate Keras MHA semantics
# ---------------------
class KerasStyleMHA(nn.Module):
    """
    Standalone equivalent of:
      tf.keras.layers.MultiHeadAttention(num_heads=32, key_dim=128, output_shape=(256,))
    called as attention(query=frame_timings_l, value=notes_l)  # key defaults to value
    """
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, num_heads: int = 32, key_dim: int = 128, val_dim: int = 128,out_dim: int = 256):
        super().__init__()
        embed_dim = num_heads * key_dim # 32 * 128 = 4096

        # Keras lets query have any last-dim; PyTorch needs query dim == embed_dim
        self.q_proj = nn.Linear(q_dim, embed_dim)
        # torch.nn.init.ones_(self.q_proj.weight)
        # torch.nn.init.zeros_(self.q_proj.bias)

        self.k_proj = nn.Linear(k_dim, key_dim)
        self.v_proj = nn.Linear(v_dim, val_dim)

        # Keys/values come from notes_l (last-dim = notes_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=val_dim,
            batch_first=True
        )

        # Keras output_shape projects attention output → out_dim (256)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, query_l, k_layer, v_layer, attn_mask=None, key_padding_mask=None):
        """
        frame_timings_l: (B, T, 1)
        notes_l:         (B, T, notes_dim)
        attn_mask:       (B, T, T) or (T, T) boolean/float mask (optional)
        key_padding_mask:(B, T) True for padded positions (optional)
        returns:
            out: (B, T, out_dim)
            attn_weights: (B, T, T) if average_attn_weights=False (default True gives (B, Tq, Tk) averaged over heads)
        """
        q = self.q_proj(query_l)  # (B, T, embed_dim)
        k = self.k_proj(k_layer)  # (B, T, embed_dim)
        v = self.v_proj(v_layer)  # (B, T, embed_dim)

        attn_out, attn_weights = self.mha(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        out = self.out_proj(attn_out)     # (B, T, out_dim)
        return out

class KerasStyleSelfMHA(nn.Module):
    """Self-attention with Keras-like signature: MHA(H, key_dim=D, output=O)."""
    def __init__(self, in_dim: int, num_heads: int, key_dim: int, out_dim: int):
        super().__init__()
        embed_dim = num_heads * key_dim
        self.in_proj = nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # self.attn_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        h = self.in_proj(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        # h = self.attn_norm(h)
        return self.out_proj(h)


# Unsure if this is needed, maybe a simple Linear would work exactly the same?
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

# ---------------------
# Models
# ---------------------
class GeneratorSmaller2(nn.Module):
    def __init__(self):
        super().__init__()
        # Frames
        self.frames_lstm1 = nn.LSTM(input_size=FRAME_LENGTH, hidden_size=128, batch_first=True)
        self.frames_lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Notes
        self.notes_proj = TimeDistributed(nn.Linear(NOTE_LENGTH, 128), batch_first=True)
        self.notes_proj_relu = nn.ReLU()
        self.notes_lstm_fwd = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.notes_bilstm1 = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=True)
        self.notes_bilstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)

        # Walls
        self.walls_proj = TimeDistributed(nn.Linear(WALL_LENGTH, 16), batch_first=True)
        self.walls_proj_relu = nn.ReLU()
        self.walls_lstm_fwd = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)
        self.walls_bilstm1 = nn.LSTM(input_size=16, hidden_size=16, batch_first=True, bidirectional=True)

        self.cross_notes = KerasStyleMHA(q_dim=1, k_dim=517, v_dim=517, num_heads=8, key_dim=16, val_dim=256, out_dim=256)
        self.cross_walls = KerasStyleMHA(q_dim=1, k_dim=37, v_dim=37, num_heads=2, key_dim=8, val_dim=8, out_dim=16)

        # Decoder
        dec_in = 1 + EXTRA_STATS + 256 + 16
        self.dec_lstm1 = nn.LSTM(input_size=dec_in, hidden_size=128, batch_first=True)
        self.dec_lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.out = TimeDistributed(nn.Linear(128, NUM_TARGET_CHANNELS), batch_first=True)

    def forward(self, frames_in, frame_times, extra_stats, notes_in, walls_in, note_time_diffs=None):
        B, T_in, _ = frames_in.shape
        T_out = frame_times.shape[1]

        extra_rep_out = extra_stats.unsqueeze(1).expand(-1, T_out, -1)

        # Frames
        x, _ = self.frames_lstm1(frames_in)
        x, (h2, c2) = self.frames_lstm2(x)

        # Notes
        n = self.notes_proj(notes_in)              # (B,50,128)
        n = self.notes_proj_relu(n)         # (B,50,128)
        n, _ = self.notes_lstm_fwd(n, (h2, c2))
        n, _ = self.notes_bilstm1(n)
        n, _ = self.notes_bilstm2(n)
        n_params = notes_in[:, :, :5]
        n = torch.cat([n, n_params], dim=-1)  # (B,50,517)

        # Walls
        w = self.walls_proj(walls_in)              
        w = self.walls_proj_relu(w)   
        w, _ = self.walls_lstm_fwd(w)
        w, _ = self.walls_bilstm1(w)
        w_params = walls_in[:, :, :5]
        w = torch.cat([w, w_params], dim=-1)  # (B,50,517)

        # Cross-attn + LN
        attn_notes = self.cross_notes(frame_times, n, n)
        attn_walls = self.cross_walls(frame_times, w, w)
        # attn = self.attn_ln(attn)

        # n = n[:, -1, :].unsqueeze(1).expand(-1, T_out, -1)
        dec_in = torch.cat([frame_times, extra_rep_out, attn_notes, attn_walls], dim=-1)
        z, _ = self.dec_lstm1(dec_in, (h2, c2))
        # z, _ = self.dec_lstm2(z)
        out = self.out(z)
        return out

class GeneratorSmaller3(nn.Module):
    def __init__(self):
        super().__init__()
        # Frames
        self.frames_mha1 = KerasStyleSelfMHA(in_dim=FRAME_LENGTH + EXTRA_STATS, num_heads=4, key_dim=32, out_dim=8)
        self.frames_mha1_proj = nn.Linear(FRAME_COUNT_INPUT * 8, 8)
        # Notes: LSTM -> self-MHA(4, key_dim=256, out=256) x2 -> concat last 5
        self.notes_proj = TimeDistributed(nn.Linear(NOTE_LENGTH, 128), batch_first=True)
        self.notes_proj_relu = nn.ReLU()
        self.notes_lstm_fwd = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.notes_mha1 = KerasStyleSelfMHA(in_dim=128, num_heads=16, key_dim=128, out_dim=128)
        self.notes_mha2 = KerasStyleSelfMHA(in_dim=128, num_heads=16, key_dim=128, out_dim=128)

        # Walls: LSTM -> concat last 5 (no pre-proj; MHA will project)
        self.walls_proj = TimeDistributed(nn.Linear(WALL_LENGTH, 16), batch_first=True)
        self.walls_proj_relu = nn.ReLU()
        self.walls_mha1 = KerasStyleSelfMHA(in_dim=16, num_heads=4, key_dim=16, out_dim=16)

        # Cross-attns
        # After concat, notes dim = 128 + 5 = 133; walls dim = 16 + 5 = 21
        
        self.cross_notes = KerasStyleMHA(q_dim=1, k_dim=141, v_dim=141, num_heads=8, key_dim=16, val_dim=256, out_dim=256)
        self.cross_walls = KerasStyleMHA(q_dim=1, k_dim=29, v_dim=29, num_heads=2, key_dim=8, val_dim=8, out_dim=16)
    
        # Decoder
        dec_in = 1 + EXTRA_STATS + 256 + 16 + 8
        self.dec_lstm1 = KerasStyleSelfMHA(in_dim=dec_in, num_heads=16, key_dim=128, out_dim=128)
        self.dec_lstm2 = KerasStyleSelfMHA(in_dim=128, num_heads=16, key_dim=128, out_dim=128)
        # self.out = nn.LSTM(input_size=128, hidden_size=NUM_TARGET_CHANNELS, batch_first=True)
        self.out = TimeDistributed(nn.Linear(128, NUM_TARGET_CHANNELS), batch_first=True)

    def forward(self, frames_in, frame_times, extra_stats, notes_in, walls_in, note_time_diffs=None):
        B, T_in, _ = frames_in.shape
        T_out = frame_times.shape[1]

        extra_rep_in = extra_stats.unsqueeze(1).expand(-1, T_in, -1)
        extra_rep_out = extra_stats.unsqueeze(1).expand(-1, T_out, -1)

        # Frames
        x = torch.cat([frames_in, extra_rep_in], dim=-1)
        x = self.frames_mha1(x)
        x = torch.Tensor.view(x, (B, -1))
        x = self.frames_mha1_proj(x)
        x_n = x.unsqueeze(1).expand(-1, NOTE_COUNT, -1)
        x_w = x.unsqueeze(1).expand(-1, WALL_COUNT, -1)
        x_f = x.unsqueeze(1).expand(-1, FRAME_COUNT_OUTPUT, -1)

        # Notes
        n = self.notes_proj(notes_in)              # (B,50,128)
        n = self.notes_proj_relu(n)         # (B,50,128)
        n, _ = self.notes_lstm_fwd(n)   # (B,50,64)
        n = self.notes_mha1(n)                            # (B,50,256)
        n = self.notes_mha2(n)                            # (B,50,256)
        n_params = notes_in[:, :, :5]                    # (B,50,5)
        n = torch.cat([n, n_params, x_n], dim=-1)              # (B,50,261)

        # Walls
        w = self.walls_proj(walls_in)              
        w = self.walls_proj_relu(w)         
        w = self.walls_mha1(w)    # (B,50,64)
        w_params = walls_in[:, :, :5]                    # (B,50,5)
        w = torch.cat([w, w_params, x_w], dim=-1)              # (B,50,69)

        notes_attn = self.cross_notes(frame_times, n, n)  # (B,50,256)
        walls_attn = self.cross_walls(frame_times, w, w)  # (B,50,32)

        dec_in = torch.cat([frame_times, extra_rep_out, notes_attn, walls_attn, x_f], dim=-1)
        # z, _ = self.dec_lstm1(dec_in)
        z = self.dec_lstm1(dec_in)
        z = self.dec_lstm2(z)
        out = self.out(z)
        return out

# ---------------------
# Sinusoidal positional encodings + stabilized attention for GeneratorSmaller4
# ---------------------
def _sinusoidal_pe(T: int, d: int, device=None, dtype=torch.float32):
    """(T, d) standard sinusoidal positional encoding (works for odd d)."""
    pos = torch.arange(T, dtype=dtype).unsqueeze(1)  # (T,1)
    half = (d + 1) // 2
    div = torch.exp(torch.arange(0, half, dtype=dtype) * -(float(np.log(10000.0)) / max(1, half)))
    pe = torch.zeros(T, d, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div[:pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe.to(device)


class KerasStyleSelfMHA2(nn.Module):
    """Self-attention with residual connection + LayerNorm (stability fix)."""
    def __init__(self, in_dim: int, num_heads: int, key_dim: int, out_dim: int):
        super().__init__()
        embed_dim = num_heads * key_dim
        self.in_proj = nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.resid = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.norm1(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        a = self.out_proj(a)
        return self.norm2(self.resid(x) + a)


class KerasStyleMHA2(nn.Module):
    """Cross-attention (Keras-style) with LayerNorm on the output."""
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, num_heads: int = 32, key_dim: int = 128, val_dim: int = 128, out_dim: int = 256):
        super().__init__()
        embed_dim = num_heads * key_dim
        self.q_proj = nn.Linear(q_dim, embed_dim)
        self.k_proj = nn.Linear(k_dim, key_dim)
        self.v_proj = nn.Linear(v_dim, val_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=key_dim, vdim=val_dim, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, query_l, k_layer, v_layer, attn_mask=None, key_padding_mask=None):
        q = self.q_proj(query_l)
        k = self.k_proj(k_layer)
        v = self.v_proj(v_layer)
        # need_weights=False: weights are discarded, and requesting them
        # blocks the fused SDPA path (materializes (B,H,Tq,Tk) tensors)
        attn_out, _ = self.mha(query=q, key=k, value=v, attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)
        return self.norm(self.out_proj(attn_out))


def _time_pe(times: torch.Tensor, d: int):
    """(B,T,d) sinusoidal positional encoding keyed to actual token times (seconds),
    scaled to ~50fps so position ~= 'frames from now'."""
    pos = (times * 50.0).unsqueeze(-1)
    half = (d + 1) // 2
    div = torch.exp(torch.arange(0, half, device=times.device, dtype=times.dtype) * -(float(np.log(10000.0)) / max(1, half)))
    pe = torch.zeros(*times.shape, d, device=times.device, dtype=times.dtype)
    pe[..., 0::2] = torch.sin(pos * div[:pe[..., 0::2].shape[-1]])
    pe[..., 1::2] = torch.cos(pos * div[:pe[..., 1::2].shape[-1]])
    return pe


def _kinematic_channels(frames: torch.Tensor):
    """First + second finite-difference derivatives (velocity/acceleration) of the
    three hilt positions, normalized by local frame dt. Layout [dt, h3, hq4, l3, lq4, r3, rq4].
    Returns (B,T,18): [vel_h3, acc_h3, vel_l3, acc_l3, vel_r3, acc_r3]."""
    dt = frames[:, :, 0:1]
    dt_step = torch.clamp(dt[:, 1:, :] - dt[:, :-1, :], min=EPS)
    dt_step = torch.cat([dt_step, dt_step[:, -1:, :]], dim=1)
    out = []
    for a, b in [(1, 3), (8, 10), (15, 17)]:
        pos = frames[:, :, a:b + 1]
        vel = (pos[:, 1:, :] - pos[:, :-1, :]) / dt_step[:, 1:, :]
        vel = torch.cat([vel, vel[:, -1:, :]], dim=1)
        acc = (vel[:, 1:, :] - vel[:, :-1, :]) / dt_step[:, 1:, :]
        acc = torch.cat([acc, acc[:, -1:, :]], dim=1)
        out += [vel, acc]
    return torch.cat(out, dim=-1)


class GeneratorSmaller4(nn.Module):
    """
    Stable variant of GeneratorSmaller3, tuned for speed maps / dense blocks:
      - sinusoidal positional encodings: TIME-based for notes/walls (position = note_time*50
        => 'lands in X frames from now'), absolute-index for frames/decoder (regular grid)
      - residual + LayerNorm in every attention block (fixes the uniform-attention collapse)
      - frames encoder widened to 128 dims, fed per-frame into the decoder
      - velocity + acceleration channels appended to the pose representation
      - small embedding MLP for raw note/wall params before cross-attention
    """
    def __init__(self):
        super().__init__()
        frame_chan = FRAME_LENGTH + 18  # + velocity/accel
        # Frames: (B,50,40)+extra -> self-MHA -> (B,50,128) per-frame context
        self.frames_mha1 = KerasStyleSelfMHA2(in_dim=frame_chan + EXTRA_STATS, num_heads=4, key_dim=32, out_dim=128)
        # Notes
        self.notes_proj = TimeDistributed(nn.Linear(NOTE_LENGTH, 128), batch_first=True)
        self.notes_proj_relu = nn.ReLU()
        self.notes_lstm_fwd = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.notes_mha1 = KerasStyleSelfMHA2(in_dim=128, num_heads=16, key_dim=128, out_dim=128)
        self.notes_mha2 = KerasStyleSelfMHA2(in_dim=128, num_heads=16, key_dim=128, out_dim=128)
        self.notes_params_proj = TimeDistributed(nn.Linear(5, 16), batch_first=True)
        self.notes_params_relu = nn.ReLU()
        # Walls
        self.walls_proj = TimeDistributed(nn.Linear(WALL_LENGTH, 16), batch_first=True)
        self.walls_proj_relu = nn.ReLU()
        self.walls_mha1 = KerasStyleSelfMHA2(in_dim=16, num_heads=4, key_dim=16, out_dim=16)
        self.walls_params_proj = TimeDistributed(nn.Linear(5, 16), batch_first=True)
        self.walls_params_relu = nn.ReLU()
        # Cross-attns (notes dim: 128+16=144; walls dim: 16+16=32; frames context 128)
        self.cross_notes = KerasStyleMHA2(q_dim=1, k_dim=144, v_dim=144, num_heads=8, key_dim=16, val_dim=256, out_dim=256)
        self.cross_walls = KerasStyleMHA2(q_dim=1, k_dim=32, v_dim=32, num_heads=2, key_dim=8, val_dim=8, out_dim=16)
        # Decoder
        dec_in = 1 + EXTRA_STATS + 256 + 16 + 128
        self.dec_lstm1 = KerasStyleSelfMHA2(in_dim=dec_in, num_heads=16, key_dim=128, out_dim=128)
        self.dec_lstm2 = KerasStyleSelfMHA2(in_dim=128, num_heads=16, key_dim=128, out_dim=128)
        self.out = TimeDistributed(nn.Linear(128, NUM_TARGET_CHANNELS), batch_first=True)
        # Positional encodings (absolute-index for the regular frames/decoder grids)
        self.register_buffer("pe_frames", _sinusoidal_pe(FRAME_COUNT_INPUT, frame_chan + EXTRA_STATS))
        self.register_buffer("pe_dec", _sinusoidal_pe(FRAME_COUNT_OUTPUT, dec_in))

    def forward(self, frames_in, frame_times, extra_stats, notes_in, walls_in, note_time_diffs=None):
        B, T_in, _ = frames_in.shape
        T_out = frame_times.shape[1]

        extra_rep_in = extra_stats.unsqueeze(1).expand(-1, T_in, -1)
        extra_rep_out = extra_stats.unsqueeze(1).expand(-1, T_out, -1)

        # Frames -> kinematics + per-frame context (B,T_in,128)
        f = torch.cat([frames_in, _kinematic_channels(frames_in)], dim=-1)
        x = torch.cat([f, extra_rep_in], dim=-1)
        x = x + self.pe_frames[:T_in]
        x = self.frames_mha1(x)
        x_f = x[:, :T_out]  # (B,50,128) per-frame past context

        # Notes (time-based PE: position = note_time*50 => 'frames from now')
        n = self.notes_proj(notes_in)
        n = self.notes_proj_relu(n)
        n = n + _time_pe(notes_in[:, :, 0], 128)
        n, _ = self.notes_lstm_fwd(n)
        n = self.notes_mha1(n)
        n = self.notes_mha2(n)
        n_params = self.notes_params_relu(self.notes_params_proj(notes_in[:, :, :5]))
        n = torch.cat([n, n_params], dim=-1)  # (B,50,144)

        # Walls
        w = self.walls_proj(walls_in)
        w = self.walls_proj_relu(w)
        w = w + _time_pe(walls_in[:, :, 0], 16)
        w = self.walls_mha1(w)
        w_params = self.walls_params_relu(self.walls_params_proj(walls_in[:, :, :5]))
        w = torch.cat([w, w_params], dim=-1)  # (B,50,32)

        notes_attn = self.cross_notes(frame_times, n, n)  # (B,50,256)
        walls_attn = self.cross_walls(frame_times, w, w)  # (B,50,16)

        dec_in = torch.cat([frame_times, extra_rep_out, notes_attn, walls_attn, x_f], dim=-1)
        dec_in = dec_in + self.pe_dec[:T_out]
        z = self.dec_lstm1(dec_in)
        z = self.dec_lstm2(z)
        out = self.out(z)
        return out


def build_model(name: str) -> nn.Module:
    if name == "smaller_2":
        return GeneratorSmaller2()
    elif name == "smaller_3":
        return GeneratorSmaller3()
    elif name == "smaller_4":
        return GeneratorSmaller4()
    else:
        raise ValueError("Unknown model name")

# A small value to prevent division by zero in normalization
EPSILON = 1e-7

def normalize_quaternion(q: torch.Tensor):
    """
    Normalizes a batch of quaternions.
    Input shape: (..., 4)
    """
    norm = torch.linalg.norm(q, dim=-1, keepdim=True) + EPSILON
    return q / norm, norm

def quaternion_to_rotation_matrix(q: torch.Tensor):
    """
    Converts a batch of quaternions to a batch of rotation matrices.
    Input shape: (..., 4)
    Output shape: (..., 3, 3)
    """
    # Unpack quaternion components
    q_x, q_y, q_z, q_w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Pre-calculate squared terms
    q_x2, q_y2, q_z2 = q_x**2, q_y**2, q_z**2
    
    # Calculate rotation matrix components
    R = q.new_zeros(q.shape[:-1] + (3, 3))
    
    R[..., 0, 0] = 1 - 2 * (q_y2 + q_z2)
    R[..., 0, 1] = 2 * (q_x * q_y - q_z * q_w)
    R[..., 0, 2] = 2 * (q_x * q_z + q_y * q_w)
    
    R[..., 1, 0] = 2 * (q_x * q_y + q_z * q_w)
    R[..., 1, 1] = 1 - 2 * (q_x2 + q_z2)
    R[..., 1, 2] = 2 * (q_y * q_z - q_x * q_w)
    
    R[..., 2, 0] = 2 * (q_x * q_z - q_y * q_w)
    R[..., 2, 1] = 2 * (q_y * q_z + q_x * q_w)
    R[..., 2, 2] = 1 - 2 * (q_x2 + q_y2)
    
    return R

def calculate_saber_tip_position(hilt_position: torch.Tensor, q: torch.Tensor, saber_length: float, rotation_length: float):
    """
    Calculates the tip and normal vectors for a batch of sabers.
    hilt_position shape: (..., 3)
    q shape: (..., 4)
    """
    rotation_matrix = quaternion_to_rotation_matrix(q)
    
    # Define base vectors on the same device as the inputs
    saber_dir_vec = torch.tensor([[0], [0], [-saber_length]], dtype=q.dtype, device=q.device)
    normal_y_vec = torch.tensor([[0], [rotation_length], [0]], dtype=q.dtype, device=q.device)
    normal_x_vec = torch.tensor([[rotation_length], [0], [0]], dtype=q.dtype, device=q.device)
    
    # Transform vectors using the rotation matrix
    # The '@' operator is used for matrix multiplication
    saber_direction = (rotation_matrix @ saber_dir_vec).squeeze(-1)
    tip_position = hilt_position + saber_direction
    
    saber_normal_y = (rotation_matrix @ normal_y_vec).squeeze(-1)
    saber_normal_x = (rotation_matrix @ normal_x_vec).squeeze(-1)
    
    return tip_position, saber_normal_y, saber_normal_x

_EPS = 1e-6

# 50-long base mask (trim/pad to T at runtime)
_BASE_MASK_50 = torch.tensor([
    16.9, 13.0, 10.0, 8.0, 6.9, 4.4, 3.1, 2.3, 1.5, 1.2,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
], dtype=torch.float32)

def _make_time_mask(frame_times: torch.Tensor) -> torch.Tensor:
    """Return (B, T) base mask from 50-long template."""
    B, T, _ = frame_times.shape
    device = frame_times.device
    base = _BASE_MASK_50.to(device)
    if T < base.numel():
        base = base[:T]
    elif T > base.numel():
        base = torch.cat([base, torch.ones(T - base.numel(), device=device)], dim=0)
    return base.unsqueeze(0).expand(B, -1)

def _normalize_quaternion(q: torch.Tensor):
    """q (..., 4) -> (unit q, norms[...,1]). Order = (x,y,z,w)."""
    norms = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(_EPS)
    return q / norms, norms

def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """q (...,4) (x,y,z,w) -> R (...,3,3)."""
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
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)
    return R

def _saber_dirs(R: torch.Tensor, saber_length: float, rotation_length: float):
    """R (...,3,3) -> tip_dir, n_y, n_x all (...,3)."""
    v_tip = torch.tensor([0.0, 0.0, -saber_length], dtype=R.dtype, device=R.device)
    v_y   = torch.tensor([0.0, rotation_length, 0.0], dtype=R.dtype, device=R.device)
    v_x   = torch.tensor([rotation_length, 0.0, 0.0], dtype=R.dtype, device=R.device)
    tip = (R @ v_tip.unsqueeze(-1)).squeeze(-1)
    n_y = (R @ v_y.unsqueeze(-1)).squeeze(-1)
    n_x = (R @ v_x.unsqueeze(-1)).squeeze(-1)
    return tip, n_y, n_x

def _calculate_saber_tip_position(hilt_pos: torch.Tensor, q: torch.Tensor,
                                  saber_length: float, rotation_length: float):
    """hilt_pos (...,3), q (...,4) -> tip_pos, n_y, n_x (all ...,3)."""
    R = _quaternion_to_rotation_matrix(q)
    tip_dir, n_y, n_x = _saber_dirs(R, saber_length, rotation_length)
    tip_pos = hilt_pos + tip_dir
    return tip_pos, n_y, n_x

@torch.no_grad()
def _prep_note_weights(frame_times, note_times, note_time_diffs, k_tip, k_pos):
    """
    frame_times:     (B, T)
    note_times:      (B, N)   (often N == T in your pipeline)
    note_time_diffs: (B, T)
    Returns (tip_mask, pos_mask) each (B, T)
    """
    device = frame_times.device
    ft = frame_times
    nd = torch.where(note_time_diffs == 0,
                     torch.tensor(6969.69, device=device, dtype=ft.dtype),
                     note_time_diffs)

    nt = torch.where(note_times == 0,
                     torch.tensor(696969.69, device=device, dtype=note_times.dtype),
                     note_times)
    
    nt = nt - nd

    abs_diff = (ft - nt.unsqueeze(1)).abs()
    lowest = abs_diff.min(dim=-1).values

    base = _make_time_mask(frame_times)
    tip = base * torch.clamp(1.0 - k_tip * lowest, min=0.3).pow(2)
    pos = base * torch.clamp(1.0 - k_pos * lowest, min=0.3).pow(2)
    
    return tip, pos

def custom_loss_with_angle_2_torch(
    y_true: torch.Tensor,             # (B, T, 21)
    y_pred: torch.Tensor,             # (B, T, 21)
    frame_times: torch.Tensor,        # (B, T)
    note_times_left: torch.Tensor,    # (B, Nl)  (pass (B, T) if that’s how you derive it)
    note_times_right: torch.Tensor,   # (B, Nr)
    note_time_diffs: torch.Tensor,    # (B, T)
    saber_length: float = 1.0,
):
    assert y_true.shape == y_pred.shape
    B, T, _ = y_true.shape

    with torch.no_grad():
        base_mask = _make_time_mask(frame_times)  # (B, T)
        tip_L, pos_L = _prep_note_weights(frame_times, note_times_left,
                                          note_time_diffs, k_tip=16.6, k_pos=6.9)
        tip_R, pos_R = _prep_note_weights(frame_times, note_times_right,
                                          note_time_diffs, k_tip=16.9, k_pos=6.9)

    # positions
    headset_pos_true = y_true[..., 0:3]
    headset_pos_pred = y_pred[..., 0:3]
    left_hilt_true   = y_true[..., 7:10]
    left_hilt_pred   = y_pred[..., 7:10]
    right_hilt_true  = y_true[..., 14:17]
    right_hilt_pred  = y_pred[..., 14:17]

    # quats (x,y,z,w) -> normalize and keep norms for penalty
    headset_q_true, _  = _normalize_quaternion(y_true[..., 3:7])
    headset_q_pred, hn = _normalize_quaternion(y_pred[..., 3:7])
    left_q_true, _     = _normalize_quaternion(y_true[..., 10:14])
    left_q_pred, ln    = _normalize_quaternion(y_pred[..., 10:14])
    right_q_true, _    = _normalize_quaternion(y_true[..., 17:21])
    right_q_pred, rn   = _normalize_quaternion(y_pred[..., 17:21])

    # norm penalty
    norm_loss = (1.0 - hn).abs().mean() * 0.2 \
              + (1.0 - ln).abs().mean() \
              + (1.0 - rn).abs().mean()

    # tips and axes
    headset_tip_true,  headset_y_true,  headset_x_true  = _calculate_saber_tip_position(
        headset_pos_true, headset_q_true, saber_length, saber_length)
    headset_tip_pred,  headset_y_pred,  headset_x_pred  = _calculate_saber_tip_position(
        headset_pos_pred, headset_q_pred, saber_length, saber_length)

    left_tip_true,  left_y_true,  left_x_true  = _calculate_saber_tip_position(
        left_hilt_true, left_q_true, saber_length, saber_length)
    left_tip_pred,  left_y_pred,  left_x_pred  = _calculate_saber_tip_position(
        left_hilt_pred, left_q_pred, saber_length, saber_length)

    right_tip_true, right_y_true, right_x_true = _calculate_saber_tip_position(
        right_hilt_true, right_q_true, saber_length, saber_length)
    right_tip_pred, right_y_pred, right_x_pred = _calculate_saber_tip_position(
        right_hilt_pred, right_q_pred, saber_length, saber_length)

    def sqdist_per_frame(a, b):  # (B, T, 3) -> (B, T)
        return ((a - b) ** 2).sum(dim=-1)

    pos_loss = (
        (sqdist_per_frame(headset_pos_true, headset_pos_pred) * base_mask).mean() * 0.2
        + (sqdist_per_frame(left_hilt_true, left_hilt_pred)   * pos_L).mean()
        + (sqdist_per_frame(right_hilt_true, right_hilt_pred) * pos_R).mean()
    )

    tip_pos_loss = (
        (sqdist_per_frame(headset_tip_true, headset_tip_pred) * base_mask).mean() * 0.15
        + (sqdist_per_frame(left_tip_true, left_tip_pred)     * tip_L).mean()
        + (sqdist_per_frame(right_tip_true, right_tip_pred)   * tip_R).mean()
    )

    angle_diff_y = (
        (sqdist_per_frame(headset_y_true, headset_y_pred) * base_mask).mean() * 0.15
        + (sqdist_per_frame(left_y_true, left_y_pred)     * tip_L).mean()
        + (sqdist_per_frame(right_y_true, right_y_pred)   * tip_R).mean()
    )

    angle_diff_x = (
        (sqdist_per_frame(headset_x_true, headset_x_pred) * base_mask).mean() * 0.15
        + (sqdist_per_frame(left_x_true, left_x_pred)     * tip_L).mean()
        + (sqdist_per_frame(right_x_true, right_x_pred)   * tip_R).mean()
    )

    loss = (norm_loss ** 2) * 0.05 + (pos_loss * 2.0) + tip_pos_loss + (angle_diff_y + angle_diff_x) * 0.15
    return loss

@torch.jit.script
def mse_loss(y_true, y_pred):
    return torch.mean((y_pred - y_true)**2)

# ---------------------
# Training / Export
# ---------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
torch.backends.cudnn.enabled = False
def export_onnx(model, export_path: str, device: torch.device):
    model.eval()
    B = 2
    frames_in = torch.zeros((B, FRAME_COUNT_INPUT, FRAME_LENGTH), dtype=torch.float32, device=device)
    frame_times = torch.zeros((B, FRAME_COUNT_OUTPUT, 1), dtype=torch.float32, device=device)
    extra_stats = torch.zeros((B, EXTRA_STATS), dtype=torch.float32, device=device)
    notes_in = torch.zeros((B, NOTE_COUNT, NOTE_LENGTH), dtype=torch.float32, device=device)
    walls_in = torch.zeros((B, WALL_COUNT, WALL_LENGTH), dtype=torch.float32, device=device)
    note_time_diffs = torch.zeros((B, NOTE_COUNT), dtype=torch.float32, device=device)

    input_names = ["frames_in", "frame_times", "extra_stats", "notes_in", "walls_in", "note_time_diffs"]
    output_names = ["pred"]
    dynamic_axes = {n: {0: "batch"} for n in input_names}
    dynamic_axes["pred"] = {0: "batch"}

    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    torch.onnx.export(
        model, (frames_in, frame_times, extra_stats, notes_in, walls_in, note_time_diffs),
        export_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False # The new dynamo=True doesn't work with LSTM and some other layers
    )
    print(f"[ONNX] Exported to: {export_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="smaller_3", choices=["smaller_2", "smaller_3", "smaller_4"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--export_path", type=str, default="./model_")
    parser.add_argument("--train_steps", type=int, default=None, help="Max steps per epoch for training")
    parser.add_argument("--val_steps", type=int, default=None, help="Max steps per epoch for validation")
    parser.add_argument("--warp_min", type=float, default=1.0, help="min time-warp scale (train only)")
    parser.add_argument("--warp_max", type=float, default=1.0, help="max time-warp scale (train only)")
    parser.add_argument("--x0_noise", type=float, default=0.0, help="max context-frame noise sigma (train only)")
    parser.add_argument("--vel_w", type=float, default=0.0, help="weight of the hilt velocity-MSE loss term")
    parser.add_argument("--vel_mode", type=str, default="sym", choices=["sym", "under"],
                        help="sym = full velocity MSE; under = punish only slower-than-target speed")
    args = parser.parse_args()

    set_seed(72583)
    device = device_from_arg(args.device)
    print(f"[Device] {device}")

    model = build_model(args.model).to(device)
    print(model)
    # for p in model.parameters():
    #     print(p)#{p.name} | {p.shape} | {p.dtype} | {p.device} | {p.numel()} | {p.requires_grad} ")
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    # Data
    if args.data_dir and os.path.isdir(args.data_dir):
        groups = list_npy_groups(args.data_dir)
        if not groups:
            raise RuntimeError(f"No matching x0s/x1s/x2s/x3s/ys under {args.data_dir}")
        n_val = max(1, int(len(groups) * 0.1))
        train_groups = groups[:-n_val]
        val_groups = groups[-n_val:]

        train_groups = train_groups[:100]
        val_groups = val_groups[:10]
        train_ds = WallsIterableDataset(train_groups, batch_size=args.batch_size, repeat=False, shuffle_files=True, shuffle_within=True,
                                        time_warp=(args.warp_min, args.warp_max), x0_noise=args.x0_noise)
        val_ds   = WallsIterableDataset(val_groups,   batch_size=args.batch_size, repeat=False, shuffle_files=False, shuffle_within=False)

        # Estimate default steps if not supplied
        total_train = estimate_total_examples(train_groups)
        total_val   = estimate_total_examples(val_groups)
        default_train_steps = max(1, math.ceil(total_train / args.batch_size))
        default_val_steps   = max(1, math.ceil(total_val   / args.batch_size))

    else:
        # Synthetic fallback
        class Synthetic(IterableDataset):
            def __init__(self, steps=100, batch=32):
                super().__init__()
                self.steps = steps; self.batch = batch
            def __iter__(self):
                rng = np.random.default_rng(1234)
                for _ in range(self.steps):
                    B = self.batch
                    x0 = torch.from_numpy(rng.standard_normal((B, FRAME_COUNT_INPUT, FRAME_LENGTH)).astype(np.float32))
                    x1 = torch.from_numpy(rng.standard_normal((B, FRAME_COUNT_OUTPUT, 1)).astype(np.float32))
                    x1e = torch.from_numpy(rng.standard_normal((B, EXTRA_STATS)).astype(np.float32))
                    x2 = torch.from_numpy(rng.standard_normal((B, NOTE_COUNT, NOTE_LENGTH)).astype(np.float32))
                    x3 = torch.from_numpy(rng.standard_normal((B, WALL_COUNT, WALL_LENGTH)).astype(np.float32))
                    tdiff = x2[:, :, 4].clone()
                    y = torch.from_numpy(rng.standard_normal((B, FRAME_COUNT_OUTPUT, NUM_TARGET_CHANNELS)).astype(np.float32))
                    yield (x0, x1, x1e, x2, x3, tdiff), y
        train_ds = Synthetic(steps=100, batch=args.batch_size)
        val_ds   = Synthetic(steps=10,  batch=args.batch_size)
        default_train_steps = 100
        default_val_steps   = 10

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=args.num_workers, pin_memory=(device.type=="cuda"), prefetch_factor=(2 if args.num_workers > 0 else None))
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=args.num_workers, pin_memory=(device.type=="cuda"), prefetch_factor=(2 if args.num_workers > 0 else None))

    train_steps = args.train_steps or default_train_steps
    val_steps   = args.val_steps   or default_val_steps

    best_val = float("inf")
    global_step = 0
    WARMUP_STEPS = 300
    start_epoch = 1
    resume_step = 0

    # Resume support: exact epoch+step resume from last.pt
    resume_ckpt_path = f"{args.export_path}last.pt"
    if os.path.isfile(resume_ckpt_path):
        try:
            _rc = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
            if _rc.get("epoch", 0) > 0 and _rc["arch"] == args.model:
                model.load_state_dict(_rc["model_state"])
                optimizer.load_state_dict(_rc["optimizer_state"])
                best_val = _rc.get("val_loss", float("inf"))
                global_step = _rc.get("global_step", 0)
                if _rc.get("epoch_done", True):
                    # epoch-end save: continue with the next epoch from step 0.
                    # (old checkpoints lack these fields and behave the same way)
                    start_epoch = _rc["epoch"] + 1
                else:
                    # mid-epoch periodic save: re-enter the same epoch, fast-forward
                    start_epoch = _rc["epoch"]
                    resume_step = int(_rc.get("step_in_epoch", 0))
                print(f"[Resume] epoch {start_epoch} at step {resume_step} "
                      f"(global {global_step}, val={best_val:.4f})")
            else:
                torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "val_loss": float("inf"), "arch": args.model, "epoch": 0, "global_step": 0,
                            "step_in_epoch": 0, "epoch_done": True,
                            "step_in_epoch": 0, "epoch_done": True},
                           resume_ckpt_path)
        except Exception as e:
            print(f"[Resume] failed to load {resume_ckpt_path} ({e}); starting fresh")
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "val_loss": float("inf"), "arch": args.model, "epoch": 0, "global_step": 0,
                        "step_in_epoch": 0, "epoch_done": True},
                       resume_ckpt_path)
    else:
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "val_loss": float("inf"), "arch": args.model, "epoch": 0, "global_step": 0,
                        "step_in_epoch": 0, "epoch_done": True},
                   resume_ckpt_path)

    def clamp_pred(p):
        """Bound predictions to physical ranges so the loss can't blow up (no inplace)."""
        p = p.float()
        return torch.cat([
            p[..., 0:3].clamp(-20, 20),    # head pos
            p[..., 3:7].clamp(-4, 4),      # head quat
            p[..., 7:10].clamp(-20, 20),   # left hilt
            p[..., 10:14].clamp(-4, 4),
            p[..., 14:17].clamp(-20, 20),  # right hilt
            p[..., 17:21].clamp(-4, 4),
        ], dim=-1)

    def compute_loss(batch, p=None):
        frames_in, frame_times, extra_stats, notes_in, walls_in, tdiff, y, *_ = batch
        if p is None:
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                pred = model(frames_in, frame_times, extra_stats, notes_in, walls_in, tdiff)
            pred = pred.float()
        else:
            pred = p
        note_times_left = notes_in[..., 0] * torch.max(notes_in[..., 8:18], dim=-1).values
        note_times_right = notes_in[..., 0] * torch.max(notes_in[..., 21:31], dim=-1).values
        with torch.autocast(device_type="cuda", enabled=False):
            base = custom_loss_with_angle_2_torch(y, clamp_pred(pred), frame_times,
                                                  note_times_left, note_times_right, tdiff, 1.0)
        if args.vel_w > 0.0:
            # Velocity loss on hilt positions: position-MSE alone is minimized by smooth
            # hovering (regression to the mean); matching finite differences forces the
            # model to reproduce actual swing speeds instead of sacrificing peaks.
            # vel_mode "under" = asymmetric: only punish being SLOWER than the target's
            # speed magnitude. Fast-but-misplaced is already corrected by the position/
            # tip terms, so nothing rewards motion away from notes -> collapse-safe.
            pc = clamp_pred(pred)
            hilt = [7, 8, 9, 14, 15, 16]
            dv_p = torch.diff(pc[..., hilt], dim=1)
            dv_y = torch.diff(y[..., hilt], dim=1)
            if args.vel_mode == "under":
                # Speed-along-target-direction deficit. NOT |dy|-|dp| (its gradient
                # dies at dp~0 -> the model freezes); this projects the model's
                # velocity onto the target's motion direction, so at rest the
                # gradient actively points along the required swing.
                sp_y = torch.linalg.vector_norm(dv_y, dim=-1, keepdim=True).clamp_min(1e-8)
                u_y = dv_y / sp_y
                proj = (dv_p * u_y).sum(dim=-1, keepdim=True)
                vterm = (torch.relu(sp_y - proj) ** 2).mean()
            else:
                vterm = ((dv_p - dv_y) ** 2).mean()
            base = base + args.vel_w * vterm
        return base

    def run_epoch(loader, steps, epoch, train=True, start_step=0):
        nonlocal best_val
        nonlocal global_step
        model.train(train)
        loss_sum = 0.0
        skipped = 0
        ff = 0
        average_loss = float("nan")
        if epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] = args.lr/10

        if train:
            optimizer.zero_grad(set_to_none=True)

        with tqdm(enumerate(loader, start=1), unit="batch") as _tqdm:
            _tqdm.set_description(f"Epoch {epoch}: ")
            for step, ((frames_in, frame_times, extra_stats, notes_in, walls_in, tdiff), y, hit_target) in _tqdm:
                if step <= start_step:
                    ff += 1  # exact resume: fast-forward already-trained batches
                    continue
                frames_in = frames_in.to(device, non_blocking=True)
                frame_times = frame_times.to(device, non_blocking=True)
                extra_stats = extra_stats.to(device, non_blocking=True)
                notes_in = notes_in.to(device, non_blocking=True)
                walls_in = walls_in.to(device, non_blocking=True)
                tdiff = tdiff.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                hit_target = hit_target.to(device, non_blocking=True)

                batch = (frames_in, frame_times, extra_stats, notes_in, walls_in, tdiff, y, hit_target)
                if train:
                    global_step += 1
                    if global_step <= WARMUP_STEPS:
                        for g in optimizer.param_groups:
                            g['lr'] = args.lr * max(1e-6, global_step / WARMUP_STEPS)
                    loss = compute_loss(batch)
                    if not torch.isfinite(loss):
                        skipped += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if step % 100 == 0:
                        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
                        if not torch.isfinite(flat).all().item():
                            ck = torch.load(f"{args.export_path}last.pt", map_location=device, weights_only=False)
                            model.load_state_dict(ck["model_state"])
                            optimizer.load_state_dict(ck["optimizer_state"])
                            print(f"[Epoch {epoch}] non-finite params -> restored last.pt, halting epoch")
                            return average_loss
                    if step % 500 == 0:
                        torch.save({"epoch": epoch, "global_step": global_step,
                                    "step_in_epoch": step, "epoch_done": False,
                                    "model_state": model.state_dict(),
                                    "optimizer_state": optimizer.state_dict(),
                                    "val_loss": best_val, "arch": args.model},
                                   f"{args.export_path}last.pt")
                        print(f"[Epoch {epoch}] periodic ckpt at step {step} (global {global_step})", flush=True)
                else:
                    with torch.no_grad():
                        loss = compute_loss(batch)
                        if not torch.isfinite(loss):
                            skipped += 1
                            continue

                loss_sum += loss.item()

                _tqdm.set_postfix(loss=loss_sum/max(1, step - skipped - ff), loss_last=loss.item())

                if step >= steps:
                    break

        n_steps = min(step, steps) - skipped - ff
        if n_steps <= 0:
            average_loss = float("nan")
            print(f"[Epoch {epoch}] WARNING: all {skipped} batches skipped (non-finite loss)")
        else:
            average_loss = loss_sum / max(1, n_steps)
        if not train:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "step_in_epoch": steps,
                "epoch_done": True,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": average_loss,
                "arch": args.model,
            }
            torch.save(ckpt, f"{args.export_path}last.pt")
            if average_loss < best_val:
                best_val = average_loss
                torch.save(ckpt, f"{args.export_path}best.pt")
                print(f"[Epoch {epoch}] new best val {average_loss:.6f}")
        return average_loss

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | steps: train={train_steps}, val={val_steps}"
              + (f" | resuming at step {resume_step}" if resume_step else ""))
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            ck = torch.load(f"{args.export_path}last.pt", map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            print(f"[Epoch {epoch}] detected non-finite params -> restored last.pt")
        tr = run_epoch(train_loader, steps=train_steps, epoch=epoch, train=True, start_step=resume_step)
        resume_step = 0
        va = run_epoch(val_loader, steps=val_steps, epoch=epoch, train=False)
        print(f"[Epoch {epoch}] train={tr:.6f} | val={va:.6f}")

    export_onnx(model, args.export_path, device)

if __name__ == "__main__":
    main()
