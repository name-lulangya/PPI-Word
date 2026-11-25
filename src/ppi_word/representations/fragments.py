"""Mean-pooled sequence windows and cosine farthest-point sampling."""
import numpy as np

def pool_residue_kmers(residue_embedding: np.ndarray, k: int = 5) -> np.ndarray:
    """Return stride-1 mean-pooled kmer embeddings with shape ``(L-k+1, D)``."""

    array = np.asarray(residue_embedding)
    if array.ndim != 2:
        raise ValueError(f"residue_embedding must be 2D, got {array.shape}")
    if k <= 0:
        raise ValueError("k must be positive")
    if array.shape[0] < k:
        raise ValueError(f"Sequence length {array.shape[0]} is shorter than k={k}")
    work = array.astype(np.float32, copy=False)
    prefix = np.concatenate(
        [np.zeros((1, work.shape[1]), dtype=np.float32), np.cumsum(work, axis=0, dtype=np.float32)],
        axis=0,
    )
    return (prefix[k:] - prefix[:-k]) / float(k)

def pool_no_overlap_residue_kmers(
    residue_embedding: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Mean-pool step-k windows with terminal tail coverage.

    Windows normally start at ``0, k, 2k, ...``.  When the terminal window
    would contain fewer than ``k`` residues, its start is shifted to
    ``length-k`` so the final word contains the last ``k`` residues and may
    overlap the preceding word.  Inputs shorter than ``k`` produce one mean
    over all available residues.
    """

    array = np.asarray(residue_embedding)
    if array.ndim != 2:
        raise ValueError(f"residue_embedding must be 2D, got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("residue_embedding must contain at least one residue")
    if k <= 0:
        raise ValueError("k must be positive")

    length = int(array.shape[0])
    starts = list(range(0, length, k))
    if starts[-1] + k > length:
        starts[-1] = max(0, length - k)
    work = array.astype(np.float32, copy=False)
    return np.stack(
        [work[start : min(start + k, length)].mean(axis=0) for start in starts],
        axis=0,
    )

def furthest_point_sample(
    embeddings: np.ndarray,
    sample_size: int = 64,
    *,
    seed: int = 42,
    device: str = "cuda:0",
) -> np.ndarray:
    """Return sorted window indices from cosine farthest-point sampling."""

    import torch

    values = np.asarray(embeddings)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"embeddings must be nonempty 2D, got {values.shape}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    count = values.shape[0]
    if count <= sample_size:
        return np.arange(count, dtype=np.int64)

    tensor = torch.from_numpy(values.astype(np.float32, copy=False)).to(device)
    normalized = torch.nn.functional.normalize(tensor, p=2, dim=1)
    torch.manual_seed(seed)
    if tensor.is_cuda:
        torch.cuda.manual_seed(seed)
    current = torch.randint(0, count, (1,), device=tensor.device)
    selected = [int(current.item())]
    distances = torch.full((count,), float("inf"), device=tensor.device)
    for _ in range(sample_size - 1):
        similarity = torch.matmul(normalized, normalized[current].transpose(0, 1)).squeeze()
        distance = 1.0 - similarity
        distances[current] = -1.0
        distances = torch.minimum(distances, distance)
        current = torch.argmax(distances).reshape(1)
        selected.append(int(current.item()))
    return np.asarray(sorted(selected), dtype=np.int64)
