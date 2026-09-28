# CholecSeg8k example data

The default demo uses **4 labeled training frames** and **4 unlabeled target-adaptation frames** from the author's locally preprocessed CholecSeg8k clips. Target frames come from the evaluation-source partition, but are used **only for adaptation**: their annotations are not included, and the default demo does not compute evaluation metrics.

## Selection

The sources are the directories and split fractions specified in:

- `working_para/working_dir_root_train_cholec_super.py`: sorted clips, first 5% for labeled training.
- `working_para/working_dir_root_eval_cholec_super.py`: sorted clips, offset 5%, remaining 95% for target images.

The available folder contains 101 clips, yielding 5 training clips and 96 target clips. We choose four clips evenly across each partition and export frame 0 from each. The partitions are explicit here, regardless of whether a legacy client passed `Starting_percentage` to its loader. `provenance.json` records clip identifiers, frame indices, and pixel hashes. Exact image hashes are checked for duplicates. This example split is not a claim of patient/video-level independence or a benchmark protocol.

## Files and colors

- `supervised.json`: four RGB PNG files with 13-channel binary `.npz` masks, including the original background channel.
- `supervised_*_mask_preview.png`: color previews of the training masks; these are not the loss targets.
- `unlabeled.json`: four RGB PNG files, with **no masks**.
- No labeled evaluation manifest is supplied.

The original converter stored BGR tensors from OpenCV. Exported PNGs have normal RGB colors for viewing, while manifests specify `color_order: BGR` so the model receives the original channel order. The model uses 224×224 inputs. Masks keep all 13 channels unchanged; no extra background channel is appended. Class names and order are included in each manifest.

The source images were already cropped/resized by the research preprocessing pipeline. This export selects frames, converts their storage channel order, and converts labels to portable arrays and previews. Regenerate with:

```bash
python scripts/export_cholec.py --source /path/to/cholecseg8k_working/output_pkl_croped \
  --output /path/to/new_example_folder
```

## Attribution and data license

CholecSeg8k is by W.-Y. Hong, C.-L. Kao, Y.-H. Kuo, J.-R. Wang, W.-L. Chang, and C.-S. Shih, based on Cholec80. Source: [official dataset](https://www.kaggle.com/datasets/newslab/cholecseg8k), [dataset paper](https://arxiv.org/abs/2012.12453), and [authors' release announcement](https://newslabntu.github.io/DanielFolio/blog/2020/CholecSeg8K/).

The authors release the dataset under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**. These derived example images, masks, and previews retain those terms; the code's MIT license does not apply to the data. The modifications are described above. No endorsement by the original dataset authors is implied.
