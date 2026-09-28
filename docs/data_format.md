# Data format

A dataset is a JSON manifest and image/mask files. Paths inside the manifest are relative to that manifest's directory:

```json
{
  "color_order": "RGB",
  "classes": ["annotated_region", "background"],
  "samples": [
    {"id": "sample_000", "image": "images/sample_000.png", "mask": "masks/sample_000.npz"}
  ]
}
```

Images are ordinary RGB PNG/JPEG files. The dataset converts them to RGB, resizes with bilinear interpolation to `image_size`, and yields float values in 0..255. Set `color_order: BGR` only when deliberately matching a checkpoint/data pipeline that used that order. The supplied Thoracic converter used RGB.

Each `.npz` contains a `masks` array of shape **C × H × W**, with only 0/1 values. Channels may overlap; do not collapse them to class IDs. Resize masks with nearest-neighbor interpolation. If including a background channel, explicitly append it as `background = (foreground.sum(axis=0) == 0)`. The Thoracic legacy examples contain one annotated channel; the exported version has that channel plus background (two total). Do not infer a class meaning from the number of channels alone.

Unlabeled manifests contain only `id` and `image`. The loader also explicitly ignores any `mask` field when running an unlabeled client. Use separate manifests for training and evaluation. Class order must be identical across labeled manifests and must match the trained dense head.

To build an archive:

```python
import numpy as np
np.savez_compressed("sample_000.npz", masks=binary_masks.astype(np.uint8))
```

## Export trusted legacy Thoracic examples

```bash
python scripts/export_thoracic.py \
  --train /path/to/train_gonogo_pkl --test /path/to/test_gonogo_pkl \
  --output examples/private_thoracic
```

The exporter handles the historical `image: 3×1×H×W`, `mask: C×1×H×W` layout, adds the background channel, and gives outputs neutral names. It writes no labels for the adaptation examples and checks exact image duplication across the selected manifests. It retains no source filenames unless `--source-map` is explicitly requested. Pickle conversion is only for trusted local data; the public runtime loader uses images and non-pickled arrays.

Choose samples and public data terms before committing clinical examples. Neutral names alone do not establish redistribution rights or remove text already embedded in image pixels.
