"""DDP training with explicit Gold, cross-species and STRING protocols.

Full runs can exceed five hours. Check-only validates metadata, and epoch states
support restarting at a completed epoch. No test table participates in selection.
Paths and task parameters come exclusively from CLI arguments/configuration.
"""
import contextlib
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from ppi_word.io import read_json, write_json
from ppi_word.models.pairwise import PairwiseScorerMLPSymmetry
from .data import Embeddings, PairDataset, StreamingPairs, collate, frame, cross_epoch_indices, accumulation_denominator


def amp(precision, device):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed-precision training requires CUDA")
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def bundle(model, config):
    return dict(format_version="ppi-word-model-1", state_dict=model.state_dict(),
                model_config={k: config[k] for k in ("temperature", "dropout")})


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, dataset, config, rank, world, device):
    p = config["training"]
    data = DataLoader(dataset, batch_size=p["batch_size"], sampler=list(range(rank, len(dataset), world)),
                      num_workers=p["workers"], collate_fn=collate)
    rows, logits, labels = [], [], []
    model.eval()
    for batch in tqdm(data, desc="validation", disable=rank != 0):
        a, b, ma, mb, y = [x.to(device) for x in batch[:5]]
        with amp(p["eval_precision"], device):
            z, _ = model(a, b, ma, mb)
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite validation logits")
        rows.extend(batch[5].tolist()); labels.extend(y.cpu().tolist()); logits.extend(z.float().cpu().tolist())
    pieces = [None]*world
    payload = (rows, labels, logits)
    if world > 1:
        dist.all_gather_object(pieces, payload)
    else:
        pieces[0] = payload
    rows, labels, logits = [np.concatenate([x[j] for x in pieces]) for j in range(3)]
    rows = rows.astype(int)
    if not np.array_equal(np.sort(rows), np.arange(len(dataset))):
        raise ValueError("Validation coverage differs")
    mask = np.ones(len(rows), bool)
    if p["selection_group"]:
        mask = dataset.frame.iloc[rows].group.to_numpy() == p["selection_group"]
    metric = roc_auc_score if p["selection_metric"] == "auroc" else average_precision_score
    result = float(metric(labels[mask], logits[mask]))
    if not np.isfinite(result):
        raise ValueError("Validation selection metric is not finite")
    return result


def run(args):
    config = read_json(args.config)
    p, task = config["training"], config["task"]
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(args.device if world == 1 else f"cuda:{os.environ['LOCAL_RANK']}")
    store = Embeddings(args.embeddings, config["prediction_windows"])
    valid = PairDataset(frame(args.valid), store)
    if task == "string":
        if not args.data_manifest:
            raise ValueError("STRING training requires the public data manifest")
        manifest = read_json(args.data_manifest)
        if not manifest.get("complete") or manifest.get("package") != "string":
            raise ValueError("Incomplete STRING input manifest")
        relative = Path(args.train).resolve().relative_to(Path(args.data_manifest).resolve().parent).as_posix()
        item = next(x for x in manifest["files"] if x["file"] == relative)
        if item["bytes"] != Path(args.train).stat().st_size:
            raise ValueError("Training CSV differs from staged manifest")
        size = int(manifest["facts"][relative]["rows"])
        train = StreamingPairs(args.train, store, rank, world)
    else:
        train = PairDataset(frame(args.train), store)
        size = len(train)
    contract = dict(config=config, world_size=world, training_rows=size, validation_rows=len(valid),
                    inputs={k: dict(path=str(Path(getattr(args, k)).resolve()), bytes=Path(getattr(args, k)).stat().st_size)
                            for k in ("train", "valid", "embeddings")})
    out = Path(args.output)
    if args.check_only:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"check.json", dict(status="checked", **contract))
        return
    if world != p["world_size"]:
        raise ValueError("Process count differs from training config; change the explicit config for a new protocol")
    if world > 1:
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/"run.json", dict(status="running", **contract))
    if world > 1:
        dist.barrier()
    seed = p["seed"] + (rank if task == "string" else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    bare = PairwiseScorerMLPSymmetry(config["temperature"], config["dropout"]).to(device)
    model = DistributedDataParallel(bare, device_ids=[device.index]) if world > 1 else bare
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"],
                                  betas=tuple(p["betas"]), eps=p["eps"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p["pos_weight"], device=device), reduction="sum")
    scaler = torch.amp.GradScaler("cuda", enabled=p["train_precision"] == "fp16")
    scheduler = None
    if p["warmup_epochs"]:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=p["warmup_start_factor"], total_iters=p["warmup_epochs"])
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"]-p["warmup_epochs"])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[p["warmup_epochs"]])
    best, stale, first_epoch, updates = float("-inf"), 0, 0, 0
    best_state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("Resume requires the same explicit input/config contract")
        bare.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        best, stale, first_epoch, updates = state["best"], state["stale"], state["epoch"]+1, state["updates"]
        best_state = state["best_state"]
        if rank == 0 and best_state is not None:
            save_state(out/"best.pt", best_state)
        rng = state["rng"][rank]
        torch.set_rng_state(rng["torch"]); random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    history = []

    def validate(epoch):
        nonlocal best, stale, best_state
        value = evaluate(bare, valid, config, rank, world, device)
        history.append(dict(epoch=epoch+1, optimizer_step=updates, validation_metric=value))
        if value > best:
            best, stale = value, 0
            if rank == 0:
                best_state = bundle(bare, config)
                best_state["state_dict"] = {k: v.detach().cpu().clone() for k, v in best_state["state_dict"].items()}
                save_state(out/"best.pt", best_state)
        else:
            stale += 1
        if rank == 0:
            write_json(out/"history.json", history)
        model.train()

    for epoch in range(first_epoch, p["epochs"]):
        if stale >= p["patience"]:
            break
        if task == "string":
            data = DataLoader(train, batch_size=p["batch_size"], num_workers=p["workers"], drop_last=True, collate_fn=collate)
            steps = size // (p["batch_size"]*world)
            local_size = steps*p["batch_size"]
        else:
            if task == "cross_species":
                indices = cross_epoch_indices(size, p["seed"], epoch, rank, world)
            elif world > 1:
                indices = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True)
                indices.set_epoch(epoch)
            else:
                indices = None
            generator = torch.Generator().manual_seed(p["seed"]+epoch*world+rank) if task == "cross_species" else None
            data = DataLoader(train, batch_size=p["batch_size"], sampler=indices, shuffle=indices is None,
                              num_workers=p["workers"], collate_fn=collate, generator=generator)
            steps, local_size = len(data), len(indices) if indices is not None else size
        if steps < 1:
            raise ValueError("No complete training minibatch")
        interval = max(1, math.ceil(steps/p["accumulation"]) // p["validations_per_epoch"])
        model.train(); optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(data, total=steps, desc=f"epoch {epoch+1}", disable=rank != 0)):
            if step >= steps:
                break
            a, b, ma, mb, y = [x.to(device) for x in batch[:5]]
            sync = (step+1) % p["accumulation"] == 0 or step+1 == steps
            context = model.no_sync() if world > 1 and not sync and task == "cross_species" else contextlib.nullcontext()
            with context:
                with amp(p["train_precision"], device):
                    logits, _ = model(a, b, ma, mb)
                    loss = criterion(logits.float() if task == "string" else logits, y)
                    denominator = accumulation_denominator(local_size, step, p["batch_size"], p["accumulation"]) if task == "cross_species" else len(y)*p["accumulation"]
                    loss = loss/denominator
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                scaler.scale(loss).backward()
            if sync:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                updates += 1
                if task == "string" and updates % interval == 0:
                    validate(epoch)
                    if stale >= p["patience"]:
                        break
        if task != "string":
            validate(epoch)
        if scheduler:
            scheduler.step()
        rng = dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                   cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        states = [None]*world
        if world > 1:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        if rank == 0:
            save_state(out/"last_epoch.pt", dict(contract=contract, epoch=epoch, updates=updates, best=best, stale=stale,
                       model=bare.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                       scheduler=scheduler.state_dict() if scheduler else None, rng=states, best_state=best_state))
    if rank == 0:
        write_json(out/"run.json", dict(status="complete", best_validation=best, **contract))
    if world > 1:
        dist.destroy_process_group()
