"""Factorized exact matching with both orientations and positional multiplicity.

Input is the public rule table, not a precomputed PPI-hit cache. A single
DuckDB scan retains rules whose endpoints occur in the supplied sequences.
Per-PPI matching returns lists of starts instead of expanding Cartesian events.
"""
from collections import defaultdict
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from ppi_word.io import ALPHABET
from .scoring import support_enrichment


def window_positions(sequence):
    positions = defaultdict(list)
    for i in range(len(sequence) - 4):
        word = sequence[i:i+5]
        if "X" not in word:
            positions[word].append(i)
    return dict(positions)


class RuleMatcher:
    """One immutable rule catalog and the endpoints required for this invocation."""
    def __init__(self, path, sequences, metadata, work, *, memory_limit, threads):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.positions = {name: window_positions(seq) for name, seq in sequences.items()}
        self.con = duckdb.connect()
        try:
            self.con.execute("SET memory_limit = ?", [memory_limit])
            self.con.execute("SET threads = ?", [threads])
            self.con.execute("SET temp_directory = ?", [str(Path(work).resolve())])
            endpoint_table = pd.DataFrame({"word": pd.Series(sorted({w for p in self.positions.values() for w in p}), dtype="string")})
            self.con.register("endpoints", endpoint_table)
            self.con.execute("""CREATE TABLE rules AS SELECT r.* FROM read_parquet(?) r
                SEMI JOIN endpoints a ON r.fragment1=a.word
                SEMI JOIN endpoints b ON r.fragment2=b.word""", [str(Path(path).resolve())])
            columns = {r[0] for r in self.con.execute("DESCRIBE rules").fetchall()}
            required = {"rule_id", "fragment1", "fragment2", "positive_ppi_support",
                        "negative_ppi_support", "enrichment", "mean_model_score"}
            if not required <= columns:
                raise ValueError(f"Missing rule columns: {required - columns}")
            bad = self.con.execute("""SELECT count(*) FROM rules WHERE rule_id IS NULL
                OR fragment1 > fragment2 OR length(fragment1)<>5 OR length(fragment2)<>5
                OR contains(fragment1,'X') OR contains(fragment2,'X')
                OR positive_ppi_support IS NULL OR negative_ppi_support IS NULL
                OR enrichment IS NULL OR mean_model_score IS NULL""").fetchone()[0]
            duplicate = self.con.execute("""SELECT count(*) FROM
                (SELECT rule_id FROM rules GROUP BY rule_id HAVING count(*)>1)""").fetchone()[0]
            duplicate_pairs = self.con.execute("""SELECT count(*) FROM
                (SELECT fragment1,fragment2 FROM rules GROUP BY ALL HAVING count(*)>1)""").fetchone()[0]
            if bad or duplicate or duplicate_pairs:
                raise ValueError("Invalid or duplicated rule identities")
            cursor = self.con.execute("""SELECT fragment1,fragment2,positive_ppi_support,
                negative_ppi_support,enrichment,mean_model_score FROM rules""")
            while batch := cursor.fetchmany(4096):
                # Bounded validation of the selected attributes, not a second full catalog scan.
                a,b,pos,neg,e,score = zip(*batch)
                if any(set(w) - ALPHABET for w in (*a,*b)):
                    raise ValueError("Invalid rule alphabet")
                expected = support_enrichment(pos,neg,metadata["n_positive_train"],metadata["n_negative_train"])
                if (not np.isfinite(score).all() or not np.allclose(expected,e,rtol=1e-12,atol=1e-12)):
                    raise ValueError("Rule attributes do not match the support scoring contract")
            self.con.unregister("endpoints")
        except BaseException:
            self.con.close()
            raise

    def match(self, protein1, protein2):
        left, right = self.positions[protein1], self.positions[protein2]
        self.con.register("left_words", pd.DataFrame({"word": pd.Series(list(left), dtype="string")}))
        self.con.register("right_words", pd.DataFrame({"word": pd.Series(list(right), dtype="string")}))
        try:
            rows = self.con.execute("""SELECT r.*,1 direction FROM rules r
                JOIN left_words a ON r.fragment1=a.word JOIN right_words b ON r.fragment2=b.word
                UNION ALL SELECT r.*,2 direction FROM rules r
                JOIN left_words a ON r.fragment2=a.word JOIN right_words b ON r.fragment1=b.word
                WHERE r.fragment1<>r.fragment2 ORDER BY rule_id,direction""").df()
        finally:
            self.con.unregister("left_words"); self.con.unregister("right_words")
        hits = []
        for row in rows.to_dict("records"):
            wa, wb = (row["fragment1"], row["fragment2"]) if row["direction"] == 1 else (row["fragment2"], row["fragment1"])
            row.update(starts1=left[wa], starts2=right[wb], occurrence_count=len(left[wa])*len(right[wb]))
            hits.append(row)
        return hits

    def close(self):
        self.con.close()
