"""Verified public checkpoint downloads and atomic local checkpoint writes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def base_manifest():
    return json.loads(Path(__file__).with_name("base_manifest.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base(directory):
    directory = Path(directory)
    for filename, expected in base_manifest()["files"].items():
        path = directory / filename
        if not path.is_file() or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Missing or invalid base checkpoint: {path}")


def download_base(directory):
    from huggingface_hub import hf_hub_download
    manifest = base_manifest()
    for filename in manifest["files"]:
        hf_hub_download(repo_id=manifest["repo_id"], revision=manifest["revision"], filename=filename, local_dir=directory)
    verify_base(directory)
    return Path(directory)


def load_base(model, directory, verify=True):
    if verify:
        verify_base(directory)
    model.load_shared({name: torch.load(Path(directory) / f"{name}.pth", map_location="cpu", weights_only=True)
                       for name in model.components})


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_task(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_shared(checkpoint["shared"])
    if model.dense_head is not None:
        if checkpoint.get("head") is None:
            raise ValueError("This checkpoint has no trained dense prediction head")
        model.dense_head.load_state_dict(checkpoint["head"], strict=True)
    return checkpoint
