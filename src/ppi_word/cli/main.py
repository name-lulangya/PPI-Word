"""Small, explicit public inference commands."""
import argparse


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(prog="ppi-word")
    commands = parser.add_subparsers(dest="command", required=True)
    embed = commands.add_parser("embed", help="ESM-2 layer-33 fragment embeddings")
    for name in ("fasta", "esm-checkpoint", "device", "output"):
        embed.add_argument("--" + name, required=True)
    embed.add_argument("--mode", choices=["no_overlap", "fps64", "stride1"], required=True)
    embed.add_argument("--precision", choices=["fp32", "bf16"], required=True)
    embed.add_argument("--storage-dtype", choices=["float32", "float16"], required=True)
    embed.add_argument("--residue-storage-dtype", choices=["float32", "float16"], required=True,
                       help="Pre-pooling representation storage: float16 for Gold/cross, float32 for STRING")
    embed.add_argument("--pairs", nargs="+", help="Embed only proteins referenced by these public pair tables")
    embed.add_argument("--shards", type=positive, default=1)
    embed.add_argument("--shard-id", type=int, default=0)
    embed.add_argument("--resume", action="store_true")
    embed.add_argument("--max-sequence-length", type=positive, default=1024,
                       help="Reject longer inputs; never truncate or split sequences")
    embed.add_argument("--pooling-method", choices=["avg_pool1d", "prefix_sum"], default="avg_pool1d",
                       help="Stride-1 mean implementation: prefix_sum follows structural/mutation analysis")
    embed.add_argument("--check-only", action="store_true")
    merge = commands.add_parser("merge-embeddings", help="Join completed disjoint shard indexes")
    merge.add_argument("--indexes", nargs="+", required=True)
    merge.add_argument("--output", required=True, help="New index.json in the parent directory containing all shards")
    for name in ("predict", "local"):
        p = commands.add_parser(name, help="PPI prediction" if name == "predict" else "Rank local fragment pairs")
        for key in ("embeddings", "pairs", "checkpoint", "config", "device", "output"):
            p.add_argument("--" + key, required=True)
        p.add_argument("--block-rows", type=positive, default=16,
                       help="16 limits intermediate pair-feature memory; does not change selected windows")
        p.add_argument("--pair-id-column", default="pair_id", help="Use complex_id for the structural complex table")
        p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                       help="FP32 is portable; use BF16 on CUDA to follow STRING scoring")
        if name == "local":
            p.add_argument("--top-k", type=positive, required=True)
    rules = commands.add_parser("rules", help="Match and score a complete support-based rule library")
    for key in ("fasta", "pairs", "rules", "metadata", "output", "memory-limit"):
        rules.add_argument("--" + key, required=True)
    rules.add_argument("--threads", type=positive, required=True)
    rules.add_argument("--pair-id-column", default="pair_id")
    train = commands.add_parser("train", help="Train a scoring network on public data")
    for key in ("embeddings", "train", "valid", "config", "device", "output"):
        train.add_argument("--"+key, required=True)
    train.add_argument("--check-only", action="store_true")
    train.add_argument("--resume", help="Trusted last_epoch.pt from the same input/config protocol")
    train.add_argument("--data-manifest", help="Required for STRING: public data manifest with authoritative row counts")
    evaluate = commands.add_parser("evaluate", help="Evaluate explicit prediction scores")
    for key in ("pairs", "predictions", "score-column", "output"):
        evaluate.add_argument("--"+key, required=True)
    evaluate.add_argument("--threshold", type=float, help="Fixed classification threshold; not selected on this table")
    for name in ("calibrate", "extract"):
        p = commands.add_parser(name, help="Calibrate or extract full stride-1 fragment pairs")
        for key in ("embeddings", "pairs", "checkpoint", "config", "device", "output", "precision"):
            p.add_argument("--"+key, required=True)
        p.add_argument("--block-rows", type=positive, default=16)
        p.add_argument("--shards", type=positive, default=1)
        p.add_argument("--shard-id", type=int, default=0)
        p.add_argument("--check-only", action="store_true")
        if name == "extract":
            p.add_argument("--policy", required=True)
        else:
            p.add_argument("--tail-probability", type=float, required=True)
    for name in ("rule-catalog", "negative-support", "rule-assemble"):
        p = commands.add_parser(name, help="Build support-based rules from selected events")
        for key in ("output", "memory-limit"):
            p.add_argument("--"+key, required=True)
        p.add_argument("--threads", type=positive, required=True)
        if name == "rule-catalog":
            p.add_argument("--events", nargs="+", required=True)
        else:
            p.add_argument("--catalog", required=True)
        if name == "negative-support":
            p.add_argument("--pairs", required=True)
            p.add_argument("--fasta", required=True)
            p.add_argument("--batch-pairs", type=positive, required=True)
            p.add_argument("--shards", type=positive, required=True)
            p.add_argument("--shard-id", type=int, required=True)
        if name == "rule-assemble":
            p.add_argument("--supports", nargs="+", required=True)
            p.add_argument("--release-id", required=True)
    args = parser.parse_args()
    if args.command == "embed":
        from .embed import run
    elif args.command == "merge-embeddings":
        from .embed import merge as run
    elif args.command == "rules":
        from .rules import run
    elif args.command == "train":
        from ppi_word.training.engine import run
    elif args.command == "evaluate":
        from .evaluate import run
    elif args.command in ("calibrate", "extract"):
        from ppi_word.rules.extraction import run
    elif args.command in ("rule-catalog", "negative-support", "rule-assemble"):
        from ppi_word.rules.construction import run
    else:
        from .model import run
    run(args)


if __name__ == "__main__":
    main()
