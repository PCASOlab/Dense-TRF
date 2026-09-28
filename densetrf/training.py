"""Base pretraining and two-client adaptation with the legacy loss schedule."""
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoints import atomic_save
from .data import ImageDataset
from .federation import wait_for
from .models.network import DenseTRF
from .visualization import save_latest_visuals


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(loader):
    while True:
        yield from loader


def objective(output, masks, step, config):
    reconstruction = output["reconstruction_loss"]
    if masks is None:
        return reconstruction, {"reconstruction": float(reconstruction.detach())}
    dense = F.binary_cross_entropy_with_logits(output["logits"], masks)
    weight = config["reconstruction_weight"] if step < config["reconstruction_until"] else 0.0
    loss = dense + weight * reconstruction if weight else dense
    return loss, {"bce": float(dense.detach()), "reconstruction": float(reconstruction.detach()), "reconstruction_weight": weight}


def optimizer_for(model, role, config):
    settings = config[role]
    parameters = list(model.initializer.parameters()) + list(model.adapter.parameters())
    parameters += list(model.processor.parameters()) + list(model.decoder.parameters())
    if role == "supervised":
        parameters += list(model.encoder.parameters()) + list(model.dense_head.parameters())
    return torch.optim.AdamW(parameters, lr=settings["lr"], weight_decay=settings["weight_decay"])


def train_client(run_dir, role, device="cpu", timeout=3600):
    if role not in ("supervised", "unlabeled"):
        raise ValueError("Role must be supervised or unlabeled")
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    config = meta["config"]
    seed_all(config["seed"] + (1 if role == "unlabeled" else 0))
    client_dir = run_dir / role
    client_dir.mkdir(exist_ok=False)
    model = DenseTRF(config["num_classes"] if role == "supervised" else None)
    model.load_shared(torch.load(run_dir / "initial.pt", map_location="cpu", weights_only=True))
    model.to(device).train()
    optimizer = optimizer_for(model, role, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=config["min_lr"])
    dataset = ImageDataset(config[role]["manifest"], labeled=role == "supervised", size=config["image_size"], num_classes=config["num_classes"])
    iterator = batches(DataLoader(dataset, batch_size=config[role]["batch_size"], shuffle=True))
    logs = client_dir / "metrics.jsonl"
    visual_every = config.get("visualization_every", 1)
    for round_id in range(meta["rounds"]):
        start = round_id * config["merge_every"]
        stop = min(start + config["merge_every"], config["steps"])
        for step in range(start, stop):
            batch = next(iterator)
            images = batch["image"].to(device)
            masks = batch["mask"].to(device) if role == "supervised" else None
            encoder_grad = role == "supervised" and config["encoder_train_start"] <= step <= config["encoder_train_end"]
            optimizer.zero_grad(set_to_none=True)
            output = model(images, encoder_grad=encoder_grad, head_detach=step < config["head_detach_until"])
            loss, metrics = objective(output, masks, step, config)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss in {role} at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5 if role == "supervised" else 0.05)
            optimizer.step()
            interval = 1000 if role == "supervised" else 10000
            if step % interval == 0:
                scheduler.step()
            with logs.open("a") as stream:
                stream.write(json.dumps({"step": step, **metrics}) + "\n")
            if visual_every > 0 and ((step + 1) % visual_every == 0 or step == config["steps"] - 1):
                save_latest_visuals(client_dir / "latest_visuals", images, output, masks=masks,
                                    sample_ids=batch["id"], color_order=dataset.color_order,
                                    class_names=dataset.class_names, step=step,
                                    max_samples=config.get("visualization_max_samples", 4))
            print(f"{role} step={step} loss={float(loss.detach()):.6f}", flush=True)
        atomic_save({"run_id": meta["id"], "round": round_id, "role": role, "shared": model.shared_state()}, client_dir / f"{round_id:06d}.pt")
        merged = wait_for(run_dir / "global" / f"{round_id:06d}.pt", timeout)
        if (merged["run_id"], merged["round"]) != (meta["id"], round_id):
            raise ValueError("Received a mismatched global checkpoint")
        model.load_shared(merged["shared"])
        # Keep optimizer moments, as the original clients did across merges.
        atomic_save({"shared": model.shared_state(), "head": model.dense_head.state_dict() if model.dense_head is not None else None,
                     "step": stop, "config": config, "role": role}, client_dir / "latest.pt")
    return client_dir / "latest.pt"


def pretrain(config, encoder_path, output, device="cpu"):
    """Train fresh slot modules on a manifest, retaining the supplied foundation encoder."""
    seed_all(config["seed"])
    model = DenseTRF()
    model.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True), strict=True)
    for component in (model.adapter, model.processor, model.decoder):
        for layer in component.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
    model.to(device).train()
    optimizer = optimizer_for(model, "base", config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=config["min_lr"])
    dataset = ImageDataset(config["base"]["manifest"], labeled=False, size=config["image_size"])
    iterator = batches(DataLoader(dataset, batch_size=config["base"]["batch_size"], shuffle=True))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for step in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        batch = next(iterator)
        images = batch["image"].to(device)
        output = model(images)
        loss = output["reconstruction_loss"]
        if not torch.isfinite(loss):
            raise ValueError("Non-finite reconstruction loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        if step % 10000 == 0:
            scheduler.step()
        visual_every = config.get("visualization_every", 1)
        if visual_every > 0 and ((step + 1) % visual_every == 0 or step == config["steps"] - 1):
            save_latest_visuals(destination / "latest_visuals", images, output,
                                sample_ids=batch["id"], color_order=dataset.color_order,
                                step=step, max_samples=config.get("visualization_max_samples", 4))
        print(f"base step={step} loss={float(loss.detach()):.6f}", flush=True)
    for name, state in model.shared_state().items():
        atomic_save(state, destination / f"{name}.pth")
    (destination / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
