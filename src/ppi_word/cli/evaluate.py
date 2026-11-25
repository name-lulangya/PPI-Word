"""Calculate metrics after exact identity alignment; no fitting or plotting."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from ppi_word.io import new_output, write_json
from ppi_word.training.data import frame


def run(args):
    pairs = frame(args.pairs)
    predictions = pd.read_csv(args.predictions, dtype={"pair_id": str, "protein1": str, "protein2": str})
    if predictions.pair_id.duplicated().any() or set(predictions.pair_id) != set(pairs.pair_id):
        raise ValueError("Prediction and label identities differ")
    z = predictions.set_index("pair_id").loc[pairs.pair_id]
    for name in ("protein1", "protein2"):
        if name in z and not np.array_equal(z[name].to_numpy(), pairs[name].to_numpy()):
            raise ValueError("Prediction endpoint identities differ")
    scores = z[args.score_column].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction scores")
    groups = {"All": np.ones(len(pairs), bool)}
    if "group" in pairs:
        groups.update({name: pairs.group.to_numpy() == name for name in sorted(pairs.group.unique())})
    rows = []
    for name, mask in groups.items():
        y, s = pairs.label.to_numpy()[mask], scores[mask]
        row = dict(group=name, n=len(y), positives=int(y.sum()),
                   AUROC=float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
                   AUPRC=float(average_precision_score(y, s)) if y.sum() else None)
        if args.threshold is not None:
            predicted = s >= args.threshold
            row.update(precision=float(precision_score(y, predicted, zero_division=0)),
                       recall=float(recall_score(y, predicted, zero_division=0)),
                       F1=float(f1_score(y, predicted, zero_division=0)))
        rows.append(row)
    out = new_output(args.output)
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    write_json(out/"run.json", dict(status="complete", rows=len(pairs), score_column=args.score_column,
                                     classification_threshold=args.threshold, AUPRC="average_precision_score"))
