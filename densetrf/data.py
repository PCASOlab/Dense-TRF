"""Portable image/one-hot-mask datasets; unlabeled loaders never open masks."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, manifest, *, labeled, size=128, num_classes=2):
        self.manifest = Path(manifest).resolve()
        spec = json.loads(self.manifest.read_text())
        self.samples = spec["samples"]
        if not self.samples:
            raise ValueError(f"Empty dataset: {manifest}")
        self.root = self.manifest.parent
        self.labeled = labeled
        self.size = size
        self.num_classes = num_classes
        self.color_order = spec.get("color_order", "RGB")
        self.class_names = spec.get("classes", [f"class {index}" for index in range(num_classes)])
        if self.color_order not in ("RGB", "BGR"):
            raise ValueError("color_order must be RGB or BGR")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        entry = self.samples[index]
        with Image.open(self.root / entry["image"]) as image:
            image = np.array(image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR))
        if self.color_order == "BGR":
            image = image[..., ::-1].copy()
        output = {"image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(), "id": entry["id"]}
        if self.labeled:
            if "mask" not in entry:
                raise ValueError(f"Labeled example {entry['id']} has no mask")
            with np.load(self.root / entry["mask"], allow_pickle=False) as archive:
                masks = archive["masks"]
            if masks.ndim != 3 or masks.shape[0] != self.num_classes:
                raise ValueError(f"Expected {self.num_classes}xHxW masks for {entry['id']}, got {masks.shape}")
            if not np.isin(masks, [0, 1]).all():
                raise ValueError("Masks must be binary, with one channel per class")
            resized = [np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (self.size, self.size), Image.Resampling.NEAREST)) for mask in masks]
            output["mask"] = torch.from_numpy(np.stack(resized)).float()
        return output
