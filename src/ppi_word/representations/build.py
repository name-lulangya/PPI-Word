"""Convert one residue matrix into explicit fragment vectors and coordinates.

Inputs: FP32 residue representations and a named window scheme. Outputs:
fragment vectors plus zero-based starts. Short sequences (<5) fail explicitly.
FPS uses the retained cosine algorithm and caller-selected device/seed.
"""
import numpy as np
from .fragments import pool_no_overlap_residue_kmers, pool_residue_kmers, furthest_point_sample


def build_fragments(residues, mode, *, storage_dtype, fps_device, fps_seed=42, pooling_method="avg_pool1d"):
    import torch
    x = np.asarray(residues)
    if x.ndim != 2 or x.shape[1] != 1280 or len(x) < 5 or not np.isfinite(x).all():
        raise ValueError("Residue embeddings must be finite (L,1280), L>=5")
    if mode == "no_overlap":
        starts = np.arange(0, len(x), 5, dtype=np.int64)
        starts[-1] = min(starts[-1], len(x) - 5)
        values = pool_no_overlap_residue_kmers(x)
    elif mode in ("stride1", "fps64"):
        starts = np.arange(len(x) - 4, dtype=np.int64)
        if pooling_method == "prefix_sum":
            values = pool_residue_kmers(x)
        elif pooling_method == "avg_pool1d":
            values = torch.nn.functional.avg_pool1d(
                torch.from_numpy(x.astype(np.float32)).T[None], 5, stride=1)[0].T.numpy()
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
    else:
        raise ValueError(f"Unknown window mode: {mode}")
    values = values.astype(storage_dtype)
    if mode == "fps64":
        selected = furthest_point_sample(values, 64, seed=fps_seed, device=fps_device)
        values, starts = values[selected], starts[selected]
    return values, starts
