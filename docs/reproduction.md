# Training protocol and implementation provenance

## Workflow

The historical workflow described by the authors is base representation pretraining (research client 11), followed by initialization of supervised client 6 and unlabeled target client 9 from the same base checkpoint. Only clients 6 and 9 participate in adaptation averaging. Clients 5/6/7 originally shared the same full concatenation model.

For shared representation parameters, the update is `theta_next = (theta_supervised + theta_unlabeled) / 2`. The dense head belongs only to the supervised client. The supervised stream can influence the unlabeled client's representation through this merge; only the unlabeled client's **local objective** is reconstruction-only.

The released checkpoint has DINOv3 ViT-B/16 (768 features), an MLP projection to 256 dimensions, 9 slots of dimension 256, 3 slot-attention iterations without the optional slot MLP, and a 4-layer 1024-wide decoder reconstructing 196 patches. The dense head consumes 256 + 768 + 9 = 1033 channels and uses a 512-wide hidden layer. The core slot/decoder computations are extracted from the VideoSAUR-derived research implementation. The DINOv3 source subset comes from upstream commit `a3a8b2f1db6a2544dfc8b376fa23df459b9f7843`.

## Inspected code settings

The provided `thoracic_adaptation.yaml` exposes the inspected settings, using zero-based step indices:

| Setting | Value |
|---|---|
| Dense input detached | steps 0–99 |
| Supervised reconstruction coefficient | 0.1 through step 399; 0 afterward |
| Foundation encoder receives supervised gradients | steps 200–300 inclusive |
| Supervised optimizer | AdamW, initial LR 1e-5, weight decay 0 |
| Unlabeled optimizer | AdamW, LR 4e-5, weight decay 1e-6 |
| Supervised / unlabeled batch sizes | 2 / 8 |
| Base optimizer | AdamW, LR 5e-5, weight decay 0 |
| Slot initialization | sampled Gaussian, including during evaluation |
| Image preprocessing | resize to configured size, then 224; `(pixel - 124) / 60` |
| SPOT patch permutation | disabled in the inspected implementation |

During steps 0–99, BCE trains the dense head while reconstruction still trains the slot representation. This is not a strictly head-only phase. After dropping the reconstruction term, BCE can still update the adapter, slots, and decoder through their concatenated outputs. The unlabeled branch retains reconstruction throughout. The feature-reconstruction target is detached from gradients.

The cosine-warm-restart scheduler is stepped at step 0 and every 1000 steps for the supervised client, and every 10000 steps for base/unlabeled training, matching the original call sites. The original printed cyclic LR was not assigned to the optimizer. The release does not reproduce that misleading printout.

## Explicit changes in this release

- Student configuration, model allocation, checkpoint loading/saving, optimizers, and dispatch are removed entirely.
- A verified, explicit base checkpoint initializes both clients. No interactive fallback or version-dependent initial loading is used.
- Merging is synchronous and atomic at the complete-snapshot level: both updates must refer to the same run and round. The old server repeatedly averaged timestamp-active snapshots without waiting for fresh updates.
- Merges occur after completed groups of updates, rather than at the original zero-based step-0 save point.
- Iteration counts are explicit and bounded. The original supervised script ended runs at epoch boundaries once both its epoch and iteration conditions were met, then restored the same initial weights for repeated runs. The default 500 steps here is an example bound, not an assertion about every historical run's length.
- Evaluation calls `.eval()` and disables gradients. The original wrapper's train/eval switch followed `Evaluation_slots` rather than `Evaluation`.
- Images use portable RGB files and binary mask arrays. Pillow resizing and explicit manifest ordering replace the legacy pickle/video loader. No automatic percentage slicing, augmentation, or hidden patient/case exclusion occurs.
- Base pretraining uses the same reconstruction architecture/objective, with a bounded image manifest. Historical mixed-pool sampling and epoch-based frame selection are not reconstructed from undocumented directory contents.

The paper describes 1000/4000/5000 iteration phases, SPOT on the base branch, and different optimization settings. This release does not silently change the inspected implementation to match those values or claim that the tiny demo reproduces published results.

## Dataset protocol

The inspected client-9 configuration uses Thoracic GoNoGo **test-distribution images without labels**, not the complement of the labeled training subset. If reproducing that experiment, declare the transductive protocol and list the target images explicitly. Target annotations must not be supplied to the unlabeled objective or used for model selection. The public loader for the unlabeled role never opens a mask even when its manifest entry includes one.

Users provide explicit training, adaptation, and evaluation manifests. The release cannot infer historical patient/video splits from differing filenames. Establish case/frame separation appropriate to the intended protocol before reporting results. The local 10-example export is only an execution sample and is not a representative clinical benchmark.

## Scope

This release covers the full image-based adaptation model, not every historical ablation or multi-backbone experiment. Image-based operation matches the inspected configurations' `Video_len = 1`. A trained segmentation head must be learned for the desired task; the public base is not a task prediction checkpoint.


## Default Cholec example

The public demo now uses the directories/split fractions from `working_dir_root_train_cholec_super.py` and `working_dir_root_eval_cholec_super.py`: the first 5% of sorted preprocessed CholecSeg8k clips supply labeled training examples, and the remaining 95% supply only unlabeled adaptation images. This adopts the intended configuration split explicitly; it does not rely on a legacy client honoring `Starting_percentage`. Four frame-0 samples are chosen from each partition. Source masks have **13** channels (not the 29-class fallback in some old client maps), so the demo head has 13 outputs. The data input size is 224, and PNG files are converted back to the legacy BGR tensor order by the manifest-driven loader. The optimizer/loss schedule remains the release adaptation example's schedule, not an evaluation-mode optimizer. No target labels or target evaluation scores are used. This small clip-based split is for execution demonstration, not a patient-level benchmark.
