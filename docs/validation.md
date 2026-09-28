# Release validation

Validated on CPU with Python 3.10.12, PyTorch 2.11.0+cu128, NumPy 1.26.4, Pillow 10.4.0, PyYAML 6.0.3, and huggingface-hub 0.36.2. CUDA execution was not exercised in the validation environment. The accepted dependency ranges are broader than this single tested environment; `requirements-tested.txt` records the versions used.

## Checks completed

- The real Hugging Face download command retrieved all five components at the pinned revision and verified their SHA-256 hashes. They match the author's local backup byte-for-byte.
- All five component state dictionaries load strictly into the standalone model.
- The extracted implementation was compared against the original DINOv3 factory and the original research SlotAttention, random initializer, decoder, and dense-head class definitions. With identical weights, inputs, and initial slots, backbone features, slots, reconstructions, masks, and dense logits each had **maximum absolute difference 0.0**. This isolates model computation from the intentionally changed portable image loader.
- Ten standard-library unit tests pass with the real checkpoint. They cover reconstruction/supervised gradient routing, the reconstruction cutoff, unlabeled loading without opening masks, slot-mask normalization, equal averaging and invalid tensor rejection, round freshness/identity checks, retention of the local head, encoder storage aliasing, the Cholec sample split/class/channel-order invariants, and latest-iteration visualization pairing and replacement without changing random state.
- A two-client CPU demo completed two updates per client, one equal-weight merge, reload into both clients, and evaluation output generation.
- A separate local Thoracic validation completed four updates per client and two merges, with shortened phase thresholds to exercise backbone fine-tuning and BCE-only training. This was an execution check, not a clinical accuracy result.
- A clean copy outside the research repository completed the full demo, confirming that the package does not import research-project modules or use the original DINOv3 source tree.
- A wheel built without network dependency resolution, installed into an isolated target directory, and loaded the real checkpoint and ran reconstruction from outside the source tree.
- A one-update base-pretraining check exercised the optional pretraining command with a supplied encoder and fresh slot modules.
- The updated default Cholec demo completed supervised and unlabeled updates and a merge with a 13-class head at 224×224 input size. It performed no scored evaluation. All eight exported images reproduce the original BGR frame tensors exactly after loading; the four exported training mask arrays match their 13-channel source labels exactly. The target-source examples have no exported masks. The Git-visible release export was checked to include these Cholec examples while excluding private Thoracic data and all downloaded weights.
- The Cholec demo also completed with visualization enabled at every iteration. Its latest supervised snapshot contains the input, prediction, ground truth, and nine slot masks; the unlabeled snapshot contains the input and slots. Saved overview and soft-slot images were visually inspected. Loss values matched the earlier run without visualization.

## Re-run

```bash
python -m densetrf download
DENSETRF_TEST_BASE=checkpoints/base python -m unittest discover -s tests -v
python scripts/run_demo.py --run outputs/new_validation_run
```

No full training experiment, paper-metric reproduction, or GPU memory claim is implied by these checks.
