"""Synchronous equal-weight merging of explicitly named participants."""
import json
import math
from pathlib import Path
import time
import uuid

import torch

from .checkpoints import atomic_save, load_base
from .models.network import DenseTRF

PARTICIPANTS = ("supervised", "unlabeled")


def average_states(states):
    if len(states) != 2:
        raise ValueError("DenseTRF adaptation requires exactly two updates")
    left, right = states
    if set(left) != set(DenseTRF.components) or set(right) != set(left):
        raise ValueError("Mismatched shared components")
    result = {}
    tensors = {}
    for component in DenseTRF.components:
        if set(left[component]) != set(right[component]):
            raise ValueError(f"Mismatched keys in {component}")
        result[component] = {}
        for name, a in left[component].items():
            b = right[component][name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Mismatched tensor: {component}.{name}")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f"Non-finite tensor: {component}.{name}")
            identity = (a.data_ptr(), b.data_ptr(), a.shape, a.stride(), a.dtype)
            if identity in tensors:
                result[component][name] = tensors[identity]
                continue
            if torch.equal(a, b):
                value = a.clone()
            elif a.is_floating_point():
                value = (a * 0.5 + b * 0.5).to(a.dtype)
            else:
                raise ValueError(f"Non-floating buffers differ: {component}.{name}")
            result[component][name] = value
            tensors[identity] = value
    return result


def initialize_run(config, run_dir, base_dir, verify=True):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    # Metadata is published only after the initial checkpoint is complete.
    torch.manual_seed(config["seed"])
    model = DenseTRF()
    load_base(model, base_dir, verify=verify)
    atomic_save(model.shared_state(), run_dir / "initial.pt")
    meta = {"id": uuid.uuid4().hex, "participants": list(PARTICIPANTS), "config": config,
            "rounds": math.ceil(config["steps"] / config["merge_every"])}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def wait_for(path, timeout):
    start = time.monotonic()
    path = Path(path)
    while not path.is_file():
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Timed out waiting for {path}; check both clients and the server")
        time.sleep(0.2)
    return torch.load(path, map_location="cpu", weights_only=True)


def serve(run_dir, timeout=3600):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    if meta["participants"] != list(PARTICIPANTS):
        raise ValueError("Unexpected participant list")
    lock = run_dir / "server.lock"
    with lock.open("x") as stream:
        stream.write(meta["id"])
    try:
        for round_id in range(meta["rounds"]):
            destination = run_dir / "global" / f"{round_id:06d}.pt"
            if destination.exists():
                raise FileExistsError("Existing round outputs: use a new experiment directory")
            updates = []
            for role in PARTICIPANTS:
                update = wait_for(run_dir / role / f"{round_id:06d}.pt", timeout)
                if (update["run_id"], update["round"], update["role"]) != (meta["id"], round_id, role):
                    raise ValueError("Update belongs to a different experiment, round, or participant")
                updates.append(update["shared"])
            merged = average_states(updates)
            atomic_save({"run_id": meta["id"], "round": round_id, "shared": merged}, destination)
            print(f"Merged round {round_id + 1}/{meta['rounds']} (supervised=0.5, unlabeled=0.5)", flush=True)
    finally:
        lock.unlink(missing_ok=True)
