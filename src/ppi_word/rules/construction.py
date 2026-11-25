"""Aggregate positive events and count negative PPI support in bounded shards.

SQL implements exact string matches and counts each rule once per PPI row.
No negative pair resampling, classification fitting, plotting or ablation.
"""
from pathlib import Path
import duckdb
import pandas as pd
from tqdm import tqdm
from ppi_word.io import read_json, write_json, read_fasta, new_output
from ppi_word.rules.matching import window_positions
from ppi_word.rules.scoring import SCORE_VERSION


def connect(args, out):
    con = duckdb.connect()
    con.execute("SET memory_limit=?", [args.memory_limit])
    con.execute("SET threads=?", [args.threads])
    con.execute("SET temp_directory=?", [str(out/"sql_work")])
    return con


def complete_shards(directories, command):
    records = [read_json(Path(d)/"run.json") for d in directories]
    if not records or any(r["status"] != "complete" or r["command"] != command for r in records):
        raise ValueError("Incomplete stage inputs")
    keys = ("source_pairs", "source_rows", "shards")
    if any(any(r[k] != records[0][k] for k in keys) for r in records):
        raise ValueError("Shards refer to different source populations")
    if sorted(r["shard_id"] for r in records) != list(range(records[0]["shards"])):
        raise ValueError("Missing or duplicate input shard")
    return records


def run(args):
    out = new_output(args.output)
    con = connect(args, out)
    try:
        if args.command == "rule-catalog":
            records = complete_shards(args.events, "extract")
            for key in ("checkpoint", "embeddings", "precision", "policy"):
                if any(r[key] != records[0][key] for r in records):
                    raise ValueError("Extraction protocols differ")
            paths = [str(Path(x)/"events.parquet") for x in args.events]
            con.execute("""CREATE TABLE candidates AS
                SELECT fragment1,fragment2,count(DISTINCT pair_id)::UBIGINT AS positive_ppi_support,
                       avg(score::DOUBLE) AS mean_model_score FROM read_parquet(?)
                WHERE NOT contains(fragment1,'X') AND NOT contains(fragment2,'X')
                GROUP BY fragment1,fragment2 HAVING count(DISTINCT pair_id)>=2""", [paths])
            con.execute("""COPY (SELECT (row_number() OVER(ORDER BY fragment1,fragment2)-1)::BIGINT rule_id,*
                FROM candidates) TO ? (FORMAT PARQUET)""", [str(out/"candidates.parquet")])
            write_json(out/"metadata.json", dict(n_positive_train=sum(r["processed_ppis"] for r in records),
                         extraction=records[0], rules=con.execute("SELECT count(*) FROM candidates").fetchone()[0]))
        elif args.command == "negative-support":
            if not 0 <= args.shard_id < args.shards:
                raise ValueError("Invalid shard ID/count")
            sequences = read_fasta(args.fasta)
            con.execute("CREATE TABLE candidates AS SELECT rule_id,fragment1,fragment2 FROM read_parquet(?)", [args.catalog])
            con.execute("CREATE TABLE counts(rule_id BIGINT, negative_ppi_support UBIGINT)")
            source_rows, processed = 0, 0
            for chunk in tqdm(pd.read_csv(args.pairs, dtype={"pair_id": str,"protein1": str,"protein2": str}, chunksize=args.batch_pairs), desc="Negative PPI support"):
                idx = range(source_rows,source_rows+len(chunk)); source_rows += len(chunk)
                keep = [i % args.shards == args.shard_id for i in idx]
                chunk = chunk.loc[keep].loc[lambda x: x.label == 0]
                if chunk.empty:
                    continue
                processed += len(chunk)
                left, right = [], []
                for row in chunk.itertuples(index=False):
                    for target, name in ((left,row.protein1),(right,row.protein2)):
                        target.extend((row.pair_id,w) for w in window_positions(sequences[name]))
                con.register("left_words", pd.DataFrame(left, columns=["pair_id","word"]))
                con.register("right_words", pd.DataFrame(right, columns=["pair_id","word"]))
                con.execute("""INSERT INTO counts SELECT rule_id,count(DISTINCT pair_id)::UBIGINT FROM (
                    SELECT l.pair_id,c.rule_id FROM left_words l JOIN candidates c ON l.word=c.fragment1
                    JOIN right_words r ON r.pair_id=l.pair_id AND r.word=c.fragment2
                    UNION
                    SELECT l.pair_id,c.rule_id FROM left_words l JOIN candidates c ON l.word=c.fragment2
                    JOIN right_words r ON r.pair_id=l.pair_id AND r.word=c.fragment1)
                    GROUP BY rule_id""")
                con.unregister("left_words"); con.unregister("right_words")
            con.execute("COPY (SELECT rule_id,sum(negative_ppi_support)::UBIGINT negative_ppi_support FROM counts GROUP BY rule_id) TO ? (FORMAT PARQUET)", [str(out/"support.parquet")])
            write_json(out/"run.json", dict(status="complete",command=args.command,source_pairs=str(Path(args.pairs).resolve()),
                       source_rows=source_rows,processed_ppis=processed,catalog=str(Path(args.catalog).resolve()),
                       shard_id=args.shard_id,shards=args.shards))
            return
        else:
            records = complete_shards(args.supports,"negative-support")
            if any(Path(r["catalog"]).resolve() != Path(args.catalog).resolve() for r in records):
                raise ValueError("Negative support catalog differs")
            meta = read_json(Path(args.catalog).parent/"metadata.json")
            npos,nneg=meta["n_positive_train"],sum(r["processed_ppis"] for r in records)
            if not npos or not nneg:
                raise ValueError("Both training classes are required")
            con.execute("CREATE TABLE negative AS SELECT rule_id,sum(negative_ppi_support)::UBIGINT negative_ppi_support FROM read_parquet(?) GROUP BY rule_id", [[str(Path(x)/"support.parquet") for x in args.supports]])
            con.execute("""CREATE TABLE final_rules AS SELECT c.rule_id,c.fragment1,c.fragment2,c.positive_ppi_support,
                coalesce(n.negative_ppi_support,0)::UBIGINT negative_ppi_support,
                ((c.positive_ppi_support+1.0)/(?+1.0))/((coalesce(n.negative_ppi_support,0)+1.0)/(?+1.0)) enrichment,
                c.mean_model_score FROM read_parquet(?) c LEFT JOIN negative n USING(rule_id)""", [npos,nneg,args.catalog])
            con.execute("COPY final_rules TO ? (FORMAT PARQUET)", [str(out/"rules.parquet")])
            write_json(out/"metadata.json", dict(format_version="ppi-word-rules-1", release_id=args.release_id,
                       complete=True,scope="complete",n_positive_train=npos,n_negative_train=nneg,score_version=SCORE_VERSION))
        write_json(out/"run.json",dict(status="complete",command=args.command))
    finally:
        con.close()
