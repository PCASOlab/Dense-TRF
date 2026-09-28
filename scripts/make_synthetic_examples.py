"""Generate deterministic, nonclinical examples for the public execution demo."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/synthetic")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:128, :128]
    for role, count in [("supervised", 4), ("unlabeled", 4), ("evaluation", 2)]:
        entries = []
        for index in range(count):
            name = f"{role}_{index:02d}"
            center = rng.integers(36, 92, size=2)
            foreground = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) < int(rng.integers(18, 32)) ** 2
            image = np.clip(rng.normal(65, 12, (128, 128, 3)), 0, 255).astype(np.uint8)
            image[foreground] = np.clip(rng.normal([170, 100, 80], 12, (foreground.sum(), 3)), 0, 255)
            Image.fromarray(image).save(output / f"{name}.png")
            entry = {"id": name, "image": f"{name}.png"}
            if role != "unlabeled":
                np.savez_compressed(output / f"{name}.npz", masks=np.stack([foreground, ~foreground]).astype(np.uint8))
                entry["mask"] = f"{name}.npz"
            entries.append(entry)
        (output / f"{role}.json").write_text(json.dumps({"color_order": "RGB", "classes": ["foreground", "background"], "samples": entries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
