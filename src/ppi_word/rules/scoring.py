"""Support enrichment and the manuscript rule score.

Supports count PPIs, while prediction multiplicities count positional matches.
All stride-1 window opportunities, including those containing X, remain in the
denominator. Only no-X rules match. Outputs are scores, not probabilities.
"""
import numpy as np

SCORE_VERSION = "support_ge2_log2p1_enrichment_x_occurrences_sqrt_original_opportunity_v1"


def support_enrichment(positive, negative, n_positive, n_negative):
    a, b = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if (a.ndim != 1 or a.shape != b.shape or n_positive <= 0 or n_negative <= 0
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a < 2) or np.any(b < 0) or np.any(a > n_positive) or np.any(b > n_negative)
            or np.any(a != np.floor(a)) or np.any(b != np.floor(b))):
        raise ValueError("Invalid PPI support counts")
    return ((a + 1) / (n_positive + 1)) / ((b + 1) / (n_negative + 1))


def contributions(enrichment, multiplicities, length1, length2):
    e, n = np.asarray(enrichment, dtype=np.float64), np.asarray(multiplicities)
    if (e.shape != n.shape or length1 < 5 or length2 < 5 or not np.isfinite(e).all()
            or np.any(e <= 0) or not np.isfinite(n).all() or np.any(n < 1)
            or np.any(n != np.floor(n))):
        raise ValueError("Invalid enrichment, multiplicity or sequence length")
    return n * (np.log1p(e) / np.log(2)) / np.sqrt((length1 - 4) * (length2 - 4))
