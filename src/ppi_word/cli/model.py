"""PPI prediction or local pair scoring from an explicit embedding index."""
import csv
import contextlib
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from ppi_word.io import read_json, write_json, read_pairs, new_output, sequence_hash
from ppi_word.models.loading import load_model


def run(args):
    index = read_json(args.embeddings)
    config = read_json(args.config)
    if index["format_version"] != "ppi-word-embeddings-1":
        raise ValueError("Unknown embedding format")
    mode = "stride1" if args.command == "local" else config["prediction_windows"]
    if index["window_mode"] != mode:
        raise ValueError(f"Expected {mode} embeddings, got {index['window_mode']}")
    pairs = read_pairs(args.pairs, index["proteins"], pair_id_column=args.pair_id_column)
    model = load_model(args.checkpoint, config, args.device)
    if args.precision == "bf16" and not str(args.device).startswith("cuda"):
        raise ValueError("BF16 scoring requires a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = new_output(args.output)
    fields = (["pair_id", "protein1", "protein2", "logit", "probability"] if args.command == "predict"
              else ["pair_id", "protein1", "protein2", "rank", "start1", "end1", "start2", "end2",
                    "fragment1", "fragment2", "local_score"])
    root = Path(args.embeddings).resolve().parent

    def load(name):
        entry = index["proteins"][name]
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Embedding file must be within the index directory")
        x = np.load(path, allow_pickle=False)
        starts = np.asarray(entry["starts"], dtype=np.int64)
        sequence = entry["sequence"]
        if (x.shape != (len(starts), 1280) or not np.isfinite(x).all() or len(starts) == 0
                or np.any(starts < 0) or np.any(starts + 5 > len(sequence))
                or len(set(starts)) != len(starts) or sequence_hash(sequence) != entry["sequence_sha256"]):
            raise ValueError(f"Embedding/coordinate mismatch: {name}")
        return torch.from_numpy(x.astype(np.float32)).to(args.device), starts, sequence

    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc=args.command):
            a, ia, sa = load(row.protein1)
            b, ib, sb = load(row.protein2)
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else contextlib.nullcontext()
            with precision:
                pa, pb = model.project_embeddings(a), model.project_embeddings(b)
                blocks = []
                for i in range(0, len(pa), args.block_rows):
                    aa = pa[i:i+args.block_rows]
                    feat = model.symmetric_pair_features(aa[:, None].expand(-1, len(pb), -1),
                        pb[None].expand(len(aa), -1, -1))
                    blocks.append(model.score_pair_features(feat)[0].cpu())
            scores = torch.cat(blocks)
            if not torch.isfinite(scores).all():
                raise ValueError(f"Nonfinite model scores: {row.pair_id}")
            base = dict(pair_id=row.pair_id, protein1=row.protein1, protein2=row.protein2)
            if args.command == "predict":
                logit = float(torch.logsumexp(scores.to(args.device) / config["temperature"], dim=(0, 1)))
                writer.writerow(dict(base, logit=logit, probability=float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64)))))
            else:
                flat = scores.float().numpy().ravel()
                # Stable score ties: ascending start1, then start2.
                order = np.lexsort((np.tile(ib, len(ia)), np.repeat(ia, len(ib)), -flat))[:args.top_k]
                for rank, pos in enumerate(order, 1):
                    i, j = divmod(int(pos), len(ib)); x, y = int(ia[i]), int(ib[j])
                    writer.writerow(dict(base, rank=rank, start1=x, end1=x+5, start2=y, end2=y+5,
                        fragment1=sa[x:x+5], fragment2=sb[y:y+5], local_score=float(flat[pos])))
    write_json(out / "run.json", dict(status="complete", pairs=len(pairs), mode=args.command,
        config=config, window_mode=mode, precision=args.precision, coordinates="zero-based, end-exclusive",
        top_k=getattr(args, "top_k", None), block_rows=args.block_rows, device=args.device,
        pair_id_column=args.pair_id_column))
