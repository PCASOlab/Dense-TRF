import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .checkpoints import load_task
from .data import ImageDataset
from .models.network import DenseTRF
from .training import seed_all


def evaluate(checkpoint, manifest, output_dir, device="cpu", threshold=0.5, seed=0):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    seed_all(seed)
    model = DenseTRF(config["num_classes"])
    load_task(model, checkpoint)
    model.to(device).eval()
    dataset = ImageDataset(manifest, labeled=True, size=config["image_size"], num_classes=config["num_classes"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    intersection = torch.zeros(config["num_classes"], dtype=torch.float64)
    union = torch.zeros_like(intersection)
    cardinality = torch.zeros_like(intersection)
    with torch.no_grad():
        for index, batch in enumerate(DataLoader(dataset, batch_size=1)):
            probabilities = model(batch["image"].to(device))["logits"].sigmoid().cpu()
            predicted = probabilities >= threshold
            truth = batch["mask"].bool()
            intersection += (predicted & truth).sum((0, 2, 3))
            union += (predicted | truth).sum((0, 2, 3))
            cardinality += predicted.sum((0, 2, 3)) + truth.sum((0, 2, 3))
            # Separate binary channels preserve overlaps, unlike argmax visualization.
            for channel in range(config["num_classes"]):
                mask = predicted[0, channel].numpy().astype(np.uint8) * 255
                Image.fromarray(mask).save(output_dir / f"sample_{index:04d}_class_{channel}.png")
    def scores(numerator, denominator):
        return [float(n / d) if d > 0 else None for n, d in zip(numerator, denominator)]
    iou = scores(intersection, union)
    dice = scores(2 * intersection, cardinality)
    result = {"iou_per_class": iou, "dice_per_class": dice,
              "mean_iou": float(np.mean([x for x in iou if x is not None])) if any(x is not None for x in iou) else None,
              "mean_dice": float(np.mean([x for x in dice if x is not None])) if any(x is not None for x in dice) else None,
              "samples": len(dataset), "threshold": threshold, "seed": seed,
              "aggregation": "global pixel counts; channels absent in both prediction and truth are excluded"}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result
