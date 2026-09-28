# Examples

The default demo uses **`cholec/`**: four labeled CholecSeg8k training frames and four unlabeled adaptation frames. The split follows the referenced Cholec configurations: first 5% of sorted clips for training, remaining 95% for adaptation. All 13 original mask channels are retained for training. The evaluation-source images are used only for adaptation; their masks are not exported and no scored evaluation runs by default.

See [Cholec sample documentation](cholec/README.md) for attribution, CC BY-NC-SA 4.0 data terms, source clip/frame provenance, and channel-order handling. View `supervised_*_mask_preview.png` for colored training masks; the loss uses the matching `.npz` files.

```bash
python scripts/run_demo.py --run outputs/cholec_demo
```

`synthetic/` remains an optional nonclinical fixture, containing 4 labeled training images, 4 unlabeled adaptation images, and 2 labeled evaluation images:

```bash
python scripts/run_demo.py --config configs/demo_synthetic.yaml --run outputs/synthetic_demo \
  --evaluation-manifest examples/synthetic/evaluation.json
```

Regenerate it with `python scripts/make_synthetic_examples.py`. These generated shapes/textures use the project's original-code license.

`private_thoracic/`, when present locally, is the previous ten-example export. It is Git-ignored and not used by the default demo. The private source mapping in `upload_artifacts/` remains excluded. To run that optional local dataset, start with `configs/demo_synthetic.yaml` (two classes) and replace the two manifest paths accordingly.
