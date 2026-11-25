"""Public embedding-index datasets and the retained task sampling policies.

Inputs are public CSV/index paths; outputs are padded fragment tensors, labels
and original row indices. No sequence filtering, negative sampling or splitting.
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from ppi_word.io import read_json


class Embeddings:
    def __init__(self, index_path, expected_mode):
        self.root = Path(index_path).resolve().parent
        self.index = read_json(index_path)
        if self.index["format_version"] != "ppi-word-embeddings-1" or self.index["window_mode"] != expected_mode:
            raise ValueError("Embedding mode/version differs from training configuration")

    def load(self, protein):
        item = self.index["proteins"][protein]
        path = (self.root / item["file"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Embedding path outside index directory")
        array = np.load(path, allow_pickle=False)
        if array.shape != (len(item["starts"]), 1280) or not np.isfinite(array).all() or len(array) == 0:
            raise ValueError(f"Invalid embedding: {protein}")
        return torch.from_numpy(array.astype(np.float32))

    def row(self, row, index):
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("Expected binary PPI label")
        return self.load(row["protein1"]), self.load(row["protein2"]), label, index


class PairDataset(Dataset):
    def __init__(self, frame, store):
        self.frame, self.store = frame, store

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return self.store.row(self.frame.iloc[index], index)


class StreamingPairs(IterableDataset):
    def __init__(self, path, store, rank, world):
        self.path, self.store, self.rank, self.world = path, store, rank, world

    def __iter__(self):
        worker = get_worker_info()
        count, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        total, current = count*self.world, count*self.rank+worker_id
        with open(self.path, newline="", encoding="utf-8") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index % total == current:
                    yield self.store.row(row, index)


def collate(rows):
    a, b, labels, indices = zip(*rows)
    aa, bb = pad_sequence(a, batch_first=True), pad_sequence(b, batch_first=True)
    ma = torch.arange(aa.shape[1])[None] < torch.tensor([len(x) for x in a])[:, None]
    mb = torch.arange(bb.shape[1])[None] < torch.tensor([len(x) for x in b])[:, None]
    return aa, bb, ma, mb, torch.tensor(labels, dtype=torch.float32), torch.tensor(indices)


def frame(path):
    data = pd.read_csv(path, dtype={"pair_id": str, "protein1": str, "protein2": str})
    if not {"pair_id", "protein1", "protein2", "label"}.issubset(data.columns):
        raise ValueError("Expected public pair table")
    if data.pair_id.duplicated().any() or not data.label.isin([0, 1]).all():
        raise ValueError("Invalid pair identities or labels")
    return data


def cross_epoch_indices(size, seed, epoch, rank, world):
    if size % world:
        raise ValueError("Cross-species training rows must divide evenly across ranks")
    generator = torch.Generator().manual_seed(seed+epoch)
    return torch.randperm(size, generator=generator).tolist()[rank::world]


def accumulation_denominator(local_size, micro_step, batch, accumulation):
    start = (micro_step // accumulation) * accumulation * batch
    return min(accumulation*batch, local_size-start)
