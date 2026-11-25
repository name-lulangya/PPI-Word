"""ESM-2 layer-33 extraction for explicitly supplied sequences."""
from pathlib import Path
import contextlib
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ppi_word.io import read_fasta, new_output, sequence_hash, write_json, read_json
from ppi_word.representations.build import build_fragments
from ppi_word.representations.esm_loading import load_esm2_representations


def run(args):
    records = read_fasta(args.fasta)
    if args.pairs:
        required = set()
        for path in args.pairs:
            for block in pd.read_csv(path, dtype=str, usecols=["protein1", "protein2"], chunksize=100000):
                required.update(block.protein1); required.update(block.protein2)
        if required - records.keys():
            raise ValueError("Pair endpoints missing from FASTA")
        records = {name: seq for name, seq in records.items() if name in required}
    if not 0 <= args.shard_id < args.shards:
        raise ValueError("Invalid embedding shard ID/count")
    records = {name: seq for i,(name,seq) in enumerate(records.items()) if i % args.shards == args.shard_id}
    if not records:
        raise ValueError("Empty embedding shard")
    if any(not 5 <= len(s) <= args.max_sequence_length for s in records.values()):
        raise ValueError(f"Supply sequences of length 5-{args.max_sequence_length}; no implicit truncation")
    if args.precision == "bf16" and not str(args.device).startswith("cuda"):
        raise ValueError("BF16 extraction requires a CUDA device")
    if not Path(args.esm_checkpoint).is_file():
        raise FileNotFoundError(args.esm_checkpoint)
    contract = dict(fasta=str(Path(args.fasta).resolve()), fasta_bytes=Path(args.fasta).stat().st_size,
                    fasta_mtime_ns=Path(args.fasta).stat().st_mtime_ns, mode=args.mode, precision=args.precision,
                    storage_dtype=args.storage_dtype, shard_id=args.shard_id, shards=args.shards,
                    residue_storage_dtype=args.residue_storage_dtype,
                    max_sequence_length=args.max_sequence_length, pooling_method=args.pooling_method,
                    proteins=list(records), checkpoint=str(Path(args.esm_checkpoint).resolve()),
                    checkpoint_bytes=Path(args.esm_checkpoint).stat().st_size, fps_device=args.device)
    out = Path(args.output) if args.resume else new_output(args.output)
    if args.check_only:
        write_json(out/"check.json",dict(status="checked",**contract))
        return
    completed = {}
    if args.resume:
        if read_json(out/"contract.json") != contract:
            raise ValueError("Embedding resume input/config differs")
        journal = out/"completed.jsonl"
        # Only an unfinished final journal line can be discarded; completed
        # records and NPY byte sizes must still agree.
        with journal.open("rb+") as handle:
            end = 0
            for line in handle:
                if not line.endswith(b"\n"):
                    break
                item = json.loads(line)
                if (out/item["file"]).stat().st_size != item["bytes"] or sequence_hash(records[item["name"]]) != item["sequence_sha256"]:
                    raise ValueError("Completed embedding changed")
                completed[item.pop("name")] = item
                end = handle.tell()
            handle.truncate(end)
    else:
        write_json(out/"contract.json",contract)
    model, alphabet = load_esm2_representations(args.esm_checkpoint)
    model = model.eval().to(args.device)
    model = model.bfloat16() if args.precision == "bf16" else model.float()
    converter = alphabet.get_batch_converter()
    index = dict(format_version="ppi-word-embeddings-1", model="esm2_t33_650M_UR50D",
                 layer=33, window_mode=args.mode, precision=args.precision,
                 residue_storage_dtype=args.residue_storage_dtype,
                 storage_dtype=args.storage_dtype, fps_device=args.device,
                 pooling_method=args.pooling_method, max_sequence_length=args.max_sequence_length,
                 fps_seed=42, proteins=completed)
    with torch.inference_mode(), (out/"completed.jsonl").open("a",encoding="utf-8",newline="\n") as journal:
        for i, (name, sequence) in enumerate(tqdm(records.items(), desc="ESM-2")):
            if name in completed:
                continue
            _, _, tokens = converter([(name, sequence)])
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else contextlib.nullcontext()
            with precision:
                rep = model(tokens.to(args.device), repr_layers=[33], return_contacts=False)["representations"][33]
            residues = rep[0, 1:len(sequence)+1].float().cpu().numpy().astype(args.residue_storage_dtype)
            values, starts = build_fragments(residues,
                args.mode, storage_dtype=args.storage_dtype, fps_device=args.device,
                pooling_method=args.pooling_method)
            filename = f"protein_{i:06d}.npy"
            np.save(out / filename, values, allow_pickle=False)
            index["proteins"][name] = dict(file=filename, bytes=(out/filename).stat().st_size, sequence=sequence,
                sequence_sha256=sequence_hash(sequence), starts=starts.tolist())
            journal.write(json.dumps(dict(name=name,**index["proteins"][name]))+"\n")
            journal.flush()
    write_json(out / "index.json", index)


def merge(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    merged, sources = None, []
    keys = ("format_version","model","layer","window_mode","precision","residue_storage_dtype","storage_dtype","fps_seed","pooling_method","max_sequence_length")
    for source in args.indexes:
        source = Path(source).resolve()
        item = read_json(source)
        if merged is None:
            merged = {k:item[k] for k in keys}; merged["proteins"]={}
        elif any(item[k] != merged[k] for k in keys):
            raise ValueError("Embedding shard protocols differ")
        for name, entry in item["proteins"].items():
            if name in merged["proteins"]:
                raise ValueError("Duplicate protein across embedding shards")
            path = (source.parent/entry["file"]).resolve()
            relative = path.relative_to(output.parent)
            merged["proteins"][name] = dict(entry,file=relative.as_posix())
        sources.append(str(source.relative_to(output.parent)))
    merged["input_indexes"] = sources
    merged["fps_device"] = "see input shard indexes"
    write_json(output, merged)
