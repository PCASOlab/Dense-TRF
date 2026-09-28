"""Convert TRUSTED local legacy Thoracic pickle files to portable examples.

Pickle may execute code: this utility is only for the owner's existing dataset.
The public training loader never unpickles example data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 pickles use _core, while some research environments run NumPy 1.
        if np.__version__.split(".")[0] == "1":
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("examples/private_thoracic"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--eval-count", type=int, default=2)
    parser.add_argument("--source-map", type=Path, help="Optional private source mapping; never commit this file")
    args = parser.parse_args()
    train = sorted(args.train.glob("*.pkl"))
    test = sorted(args.test.glob("*.pkl"))
    if len(train) < args.count or len(test) < args.count + args.eval_count:
        raise ValueError("Not enough samples for disjoint example manifests")
    args.output.mkdir(parents=True, exist_ok=False)
    selections = [("supervised", train[:args.count]), ("unlabeled", test[:args.count]),
                  ("evaluation", test[args.count:args.count + args.eval_count])]
    source_map = {}
    hashes = set()
    for role, paths in selections:
        entries = []
        for index, path in enumerate(paths):
            with path.open("rb") as stream:
                sample = LegacyUnpickler(stream).load()
            image = sample["image"]
            if image.ndim != 4 or image.shape[:2] != (3, 1):
                raise ValueError("This exporter expects 3x1xHxW still images")
            rgb = image[:, 0].transpose(1, 2, 0).astype(np.uint8)
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError("Duplicate image across example selections; select different source files")
            hashes.add(digest)
            name = f"{role}_{index:03d}"
            Image.fromarray(rgb).save(args.output / f"{name}.png")
            entry = {"id": name, "image": f"{name}.png"}
            if role != "unlabeled":
                masks = (np.asarray(sample["mask"])[:, 0] > 0).astype(np.uint8)
                background = (masks.sum(axis=0, keepdims=True) == 0).astype(np.uint8)
                masks = np.concatenate([masks, background], axis=0)
                if masks.shape[0] != 2:
                    raise ValueError("Expected one annotated region channel plus background")
                np.savez_compressed(args.output / f"{name}.npz", masks=masks)
                entry["mask"] = f"{name}.npz"
            entries.append(entry)
            source_map[name] = str(path.resolve())
        spec = {"color_order": "RGB", "classes": ["annotated_region", "background"], "samples": entries}
        (args.output / f"{role}.json").write_text(json.dumps(spec, indent=2) + "\n")
    if args.source_map:
        args.source_map.parent.mkdir(parents=True, exist_ok=True)
        args.source_map.write_text(json.dumps(source_map, indent=2) + "\n")


if __name__ == "__main__":
    main()
