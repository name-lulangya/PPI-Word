"""Explicit FASTA, pair-table and JSON contracts for public inference.

No sequence replacement, truncation, deduplication or hidden path discovery.
Input identifiers are preserved; file names for arrays are generated separately.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBOUXZ")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_fasta(path):
    records = {}
    name, parts = None, []

    def finish():
        if name is None:
            return
        sequence = "".join(parts)
        if name in records or not sequence or set(sequence) - ALPHABET:
            raise ValueError(f"Duplicate ID, empty sequence or unsupported residues: {name}")
        records[name] = sequence

    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                fields = line[1:].split()
                if not fields:
                    raise ValueError("Empty FASTA ID")
                name, parts = fields[0], []
            else:
                if name is None:
                    raise ValueError("Sequence before FASTA header")
                parts.append(line)
    finish()
    if not records:
        raise ValueError("Empty FASTA")
    return records


def read_pairs(path, sequence_ids, *, pair_id_column="pair_id"):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if pair_id_column != "pair_id":
        if pair_id_column not in frame or "pair_id" in frame:
            raise ValueError("Explicit pair-ID column is missing or conflicts with pair_id")
        frame = frame.rename(columns={pair_id_column: "pair_id"})
    required = {"pair_id", "protein1", "protein2"}
    if not required <= set(frame) or frame.empty:
        raise ValueError("Pairs require nonempty pair_id, protein1 and protein2 columns")
    if frame.pair_id.duplicated().any() or frame[list(required)].eq("").any().any():
        raise ValueError("Pair IDs must be unique and required fields nonempty")
    missing = (set(frame.protein1) | set(frame.protein2)) - set(sequence_ids)
    if missing:
        raise ValueError(f"Missing sequences/embeddings: {sorted(missing)[:10]}")
    return frame


def new_output(path):
    """An existing output directory is never silently overwritten."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=False)
    return out
