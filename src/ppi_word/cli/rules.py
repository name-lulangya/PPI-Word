"""Score new sequence pairs against an explicit support-based rule catalog."""
import csv
import json
from tqdm import tqdm
from ppi_word.io import read_fasta, read_pairs, read_json, write_json, new_output
from ppi_word.rules.matching import RuleMatcher
from ppi_word.rules.scoring import SCORE_VERSION, contributions


def run(args):
    sequences = read_fasta(args.fasta)
    pairs = read_pairs(args.pairs, sequences, pair_id_column=args.pair_id_column)
    needed = set(pairs.protein1) | set(pairs.protein2)
    sequences = {k: v for k,v in sequences.items() if k in needed}
    if any(len(seq) < 5 for seq in sequences.values()):
        raise ValueError("Rule matching requires sequences of length >=5")
    meta = read_json(args.metadata)
    if (meta.get("format_version") != "ppi-word-rules-1" or meta.get("score_version") != SCORE_VERSION
            or meta.get("scope") != "complete" or not meta.get("complete")):
        raise ValueError("A completed full-library release with the current score version is required")
    out = new_output(args.output)
    matcher = RuleMatcher(args.rules, sequences, meta, out / "work",
                          memory_limit=args.memory_limit, threads=args.threads)
    hit_fields = ["pair_id", "rule_id", "fragment1", "fragment2", "direction", "starts1", "starts2",
                  "occurrence_count", "enrichment", "mean_model_score", "score_contribution"]
    try:
        with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as pred, \
             (out / "matches.csv").open("w", newline="", encoding="utf-8") as match:
            pw = csv.DictWriter(pred, fieldnames=["pair_id", "protein1", "protein2", "rule_score",
                "matched_rule_count", "matched_occurrence_count", "pair_opportunity"])
            hw = csv.DictWriter(match, fieldnames=hit_fields)
            pw.writeheader(); hw.writeheader()
            for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="Rule prediction"):
                hits = matcher.match(row.protein1, row.protein2)
                l1, l2 = len(sequences[row.protein1]), len(sequences[row.protein2])
                weights = contributions([h["enrichment"] for h in hits], [h["occurrence_count"] for h in hits], l1, l2)
                for hit, weight in zip(hits, weights):
                    record = {k: hit[k] for k in hit_fields if k in hit}
                    record.update(pair_id=row.pair_id, score_contribution=float(weight),
                        starts1=json.dumps(hit["starts1"]), starts2=json.dumps(hit["starts2"]))
                    hw.writerow(record)
                pw.writerow(dict(pair_id=row.pair_id, protein1=row.protein1, protein2=row.protein2,
                    rule_score=float(weights.sum()), matched_rule_count=len({h["rule_id"] for h in hits}),
                    matched_occurrence_count=sum(h["occurrence_count"] for h in hits),
                    pair_opportunity=(l1-4)*(l2-4)))
    finally:
        matcher.close()
    write_json(out / "run.json", dict(status="complete", pairs=len(pairs), score_version=SCORE_VERSION,
        rule_release=meta.get("release_id"), memory_limit=args.memory_limit, threads=args.threads,
        coordinates="zero-based starts; each window has length 5"))
