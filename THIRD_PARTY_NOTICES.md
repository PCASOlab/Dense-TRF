# Third-party notices

## DINOv3

The minimal inference sources under `densetrf/_vendor/dinov3/` come from Meta's DINOv3 repository, commit `a3a8b2f1db6a2544dfc8b376fa23df459b9f7843`. Absolute package imports were redirected under the DenseTRF namespace; the layers package exports were reduced to the inference subset. Source copyright headers are retained. These files remain governed by [the DINOv3 License Agreement](licenses/DINOv3.md).

Source: https://github.com/facebookresearch/dinov3

The downloaded `encoder.pth` includes DINOv3 weights and remains subject to the applicable upstream terms. The original-code MIT license does not relicense Meta's code or weights. The release loader uses the author's complete encoder checkpoint and does not request another gated upstream weight download.

## VideoSAUR / DINOSAUR-style slots

The slot-attention implementation and random-initialization/spatial-decoder computations derive from the VideoSAUR-based research code. Copyright (c) 2023 Maximilian Seitzer and Andrii Zadaianchuk. The [MIT notice](licenses/VideoSAUR.txt) is retained. The release removes general model factories, recurrent/video wrappers, optional slot MLPs, and unrelated training dependencies while preserving the active image model's computations and state-dict names.

Source: https://github.com/martius-lab/videosaur

## Original DenseTRF contributions

The projection/concatenation task architecture, release CLI, dataset interface, checkpoint handling, and averaging integration are distributed under the root MIT license. Clinical sample data, when released, requires its own stated terms. The bundled paper is provided as the authors' manuscript and is not relicensed as software.


## CholecSeg8k example data

The images, masks, and mask previews in `examples/cholec/` derive from CholecSeg8k by W.-Y. Hong, C.-L. Kao, Y.-H. Kuo, J.-R. Wang, W.-L. Chang, and C.-S. Shih, based on Cholec80. They retain [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) terms, as stated in the [authors' release announcement](https://newslabntu.github.io/DanielFolio/blog/2020/CholecSeg8K/) and [dataset paper](https://arxiv.org/abs/2012.12453). These data are not MIT-licensed. Selection, crop/resize history, channel-order conversion, and portable mask conversion are documented in `examples/cholec/README.md` and `provenance.json`.
