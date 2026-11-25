"""Score full stride-1 pairs for calibration or bounded positive extraction.

Inputs are real pair tables and complete stride-1 embeddings. Outputs are
weighted score histograms or selected positional events. Pair-row shards have
independent outputs; completed shards are reusable after an interrupted job.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from ppi_word.io import read_json, write_json, new_output
from ppi_word.models.loading import load_model
from ppi_word.training.data import Embeddings


def selected_indices(scores, threshold, lower_bound, cap):
    """Return the locked rank prefix; ties use ascending endpoint positions."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite nonempty local-score matrix")
    if cap < lower_bound or lower_bound < 1:
        raise ValueError("Invalid selection bounds")
    n = min(values.size, cap, max(lower_bound, int((values >= threshold).sum())))
    return np.argsort(-values.ravel(), kind="stable")[:n]


def weighted_tail_threshold(scores, counts, target):
    """Smallest upper-tail prefix reaching target mass; retain threshold ties."""
    scores, counts = np.asarray(scores), np.asarray(counts, dtype=float)
    if not 0 < target < 1 or not len(scores) or np.any(counts <= 0):
        raise ValueError("Invalid weighted histogram or tail target")
    order = np.argsort(-scores)
    location = np.searchsorted(np.cumsum(counts[order]), target*counts.sum(), side="left")
    return float(scores[order[min(location, len(scores)-1)]])


@torch.no_grad()
def score_pair(model, a, b, device, precision, block_rows):
    context = (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16"
               else torch.autocast(device.type, enabled=False))
    with context:
        aa, bb = model.project_embeddings(a.to(device)), model.project_embeddings(b.to(device))
        blocks = []
        for i in range(0, len(aa), block_rows):
            x = aa[i:i+block_rows]
            f = model.symmetric_pair_features(x[:, None].expand(-1, len(bb), -1), bb[None].expand(len(x), -1, -1))
            blocks.append(model.score_pair_features(f)[0].float().cpu().numpy())
    scores = np.concatenate(blocks)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite local scores")
    return scores


def run(args):
    config = read_json(args.config)
    store = Embeddings(args.embeddings, "stride1")
    device = torch.device(args.device)
    if args.precision not in ("fp32", "bf16"):
        raise ValueError("Scoring precision must be fp32 or bf16")
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 scoring requires CUDA")
    if not 0 <= args.shard_id < args.shards:
        raise ValueError("Invalid shard ID/count")
    if args.command=="calibrate" and args.shards!=1:
        raise ValueError("Calibration uses one complete weighted sample")
    out = new_output(args.output)
    if args.check_only:
        header = pd.read_csv(args.pairs, nrows=0)
        if not {"pair_id", "protein1", "protein2", "label"}.issubset(header):
            raise ValueError("Expected a public pair table")
        write_json(out/"check.json", dict(status="checked", pairs=str(Path(args.pairs).resolve()),
                                          embeddings=str(Path(args.embeddings).resolve()), shard_id=args.shard_id, shards=args.shards))
        return
    model = load_model(args.checkpoint, config, device)
    schema = pa.schema([("pair_id", pa.string()), ("start1", pa.int32()), ("start2", pa.int32()),
                        ("fragment1", pa.string()), ("fragment2", pa.string()), ("direction", pa.int8()), ("score", pa.float32())])
    histogram = defaultdict(float)
    offset, processed = 0, 0
    writer = pq.ParquetWriter(out/"events.parquet", schema) if args.command == "extract" else None
    policy = read_json(args.policy) if args.command == "extract" else None
    try:
        for chunk in tqdm(pd.read_csv(args.pairs, dtype={"pair_id": str, "protein1": str, "protein2": str}, chunksize=1000), desc=args.command):
            indices = np.arange(offset, offset+len(chunk)); offset += len(chunk)
            chunk = chunk.loc[(indices % args.shards == args.shard_id) & (chunk.label == (1 if writer else 0))]
            events = []
            for row in chunk.itertuples(index=False):
                scores = score_pair(model, store.load(row.protein1), store.load(row.protein2), device, args.precision, args.block_rows)
                processed += 1
                if writer:
                    sequences = [store.index["proteins"][x]["sequence"] for x in (row.protein1, row.protein2)]
                    for name in (row.protein1, row.protein2):
                        entry = store.index["proteins"][name]
                        if entry["starts"] != list(range(len(entry["sequence"])-4)):
                            raise ValueError("Extraction requires the complete stride-1 window space")
                    for value in selected_indices(scores, policy["threshold"], policy["lower_bound"], policy["cap"]):
                        i, j = divmod(int(value), scores.shape[1])
                        x, y = sequences[0][i:i+5], sequences[1][j:j+5]
                        a, b = sorted((x, y))
                        events.append(dict(pair_id=row.pair_id, start1=i, start2=j, fragment1=a, fragment2=b,
                                           direction=1 if x <= y else 2, score=float(scores[i, j])))
                else:
                    weight = float(row.ppi_sampling_weight)
                    if not np.isfinite(weight) or weight <= 0:
                        raise ValueError("Calibration requires a positive inverse sampling fraction")
                    values, counts = np.unique(scores, return_counts=True)
                    for value, count in zip(values, counts):
                        histogram[float(value)] += int(count)*weight
            if writer and events:
                writer.write_table(pa.Table.from_pylist(events, schema=schema))
    finally:
        if writer:
            writer.close()
    if not processed:
        raise ValueError("No eligible pair rows")
    if not writer:
        threshold = weighted_tail_threshold(list(histogram), list(histogram.values()), args.tail_probability)
        pd.DataFrame(sorted(histogram.items()), columns=["score", "weighted_count"]).to_parquet(out/"histogram.parquet", index=False)
        write_json(out/"policy.json", dict(threshold=threshold, lower_bound=64, cap=512, target_tail_probability=args.tail_probability))
    write_json(out/"run.json", dict(status="complete", command=args.command, processed_ppis=processed,
        source_rows=offset, source_pairs=str(Path(args.pairs).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
        embeddings=str(Path(args.embeddings).resolve()), precision=args.precision, policy=policy,
        shard_id=args.shard_id, shards=args.shards))
