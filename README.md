<div align="center">
<h2> Any Resolution Any Geometry: From Multi-View To Multi-Patch </h2>

[![Project Website](https://img.shields.io/badge/Project-Website-1f6feb?logo=googlechrome&logoColor=white)](https://dreamaker-mrc.github.io/Any-Resolution-Any-Geometry/) [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.02284) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<a href="https://dreamaker-mrc.github.io/">Wenqing Cui</a><sup>1</sup>, <a href="https://zhyever.github.io/">Zhenyu Li</a><sup>1,†</sup>, Mykola Lavreniuk<sup>2,†</sup>, Jian Shi<sup>1</sup>, Ramzi Idoughi<sup>1</sup>, Xiangjun Tang<sup>1</sup>, <a href="https://peterwonka.net/">Peter Wonka</a><sup>1</sup>.
<br><sup>1</sup>KAUST, <sup>2</sup>Space Research Institute NASU-SSAU
<br><sup>†</sup>Equal contribution

</div>

## ✨ **NEWS**

- 2026-02-27: Initially release
- 2026-02-20: Accepted to CVPR 2026.



## **Environment Setup**

**Requirements:** Python ≥ 3.10, CUDA 12.4

```bash
conda create -n urgt python=3.10 -y
conda activate urgt
pip install -r requirements.txt
```


## **Pre-Train Model**

Download checkpoints from HuggingFace:

**[https://huggingface.co/Kingslanding/Any-Resolution-Any-Geometry/tree/main](https://huggingface.co/Kingslanding/Any-Resolution-Any-Geometry/tree/main)**

| Checkpoint | Training data | Recommended use |
|---|---|---|
| `ckpt_best.pth` | U4K dataset | U4K benchmark evaluation |
| `ckpt_promask_best.pth` | U4K dataset with PRO model masks | Zero-shot evaluation |

Place the downloaded checkpoints under `work_dir/ckpts/`:

```
work_dir/
└── ckpts/
    ├── ckpt_best.pth
    └── ckpt_promask_best.pth
```

You can also download directly from the command line:

```bash
mkdir -p work_dir/ckpts
huggingface-cli download Kingslanding/Any-Resolution-Any-Geometry \
    ckpt_best.pth ckpt_promask_best.pth \
    --local-dir work_dir/ckpts
```


## **User Inference**

The inference script runs the full URGT pipeline on a single image:
1. **Depth Anything v2** → coarse relative depth
2. **Metric3D v2** → coarse surface normals
3. **URGT refiner** → high-resolution refined depth + normals

### Basic usage

```bash
python tools/infer.py \
    --image    path/to/image.jpg \
    --checkpoint work_dir/ckpts/ckpt_best.pth \
    --output-dir ./output
```

### With pre-computed coarse predictions

If you already have coarse depth/normal maps (`.npy`), pass them directly to skip Steps 1 and 2:

```bash
python tools/infer.py \
    --image         path/to/image.jpg \
    --checkpoint    work_dir/ckpts/ckpt_best.pth \
    --coarse-depth  path/to/coarse_depth.npy \
    --coarse-normal path/to/coarse_normal.npy \
    --output-dir    ./output
```

### Example (provided test image)

```bash
python tools/infer.py \
    --image work_dir/examples/lab_8k.jpg \
    --checkpoint work_dir/ckpts/ckpt_best.pth \
    --output-dir work_dir/examples/output \
    --save-intermediates
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to input RGB image (JPG/PNG) |
| `--checkpoint` | *(required)* | Path to URGT checkpoint (`.pth`) |
| `--output-dir` | same dir as image | Directory to save results |
| `--save-intermediates` | off | Also save coarse depth/normal visualisations |
| `--dav2-encoder` | `vitl` | Depth Anything v2 encoder: `vits` / `vitb` / `vitl` / `vitg` |
| `--coarse-depth` | `None` | Pre-computed coarse depth (`.npy`, shape `[H, W]`); skips DAv2 |
| `--metric3d-model` | `ViT-Small` | Metric3D v2 variant: `ViT-Small` / `ViT-Large` / `ViT-giant2` |
| `--coarse-normal` | `None` | Pre-computed coarse normal (`.npy`, shape `[H, W, 3]`); skips Metric3D |
| `--patch-split` | `8 8` | Patch grid `N_H N_W`; image is resized to be divisible by these values |
| `--min-depth` | `0.001` | Minimum depth value in metres |
| `--max-depth` | `80.0` | Maximum depth value in metres |
| `--device` | auto | `cuda` or `cpu` (defaults to CUDA when available) |

### Outputs

For an input named `image.jpg`, the following files are written to `--output-dir`:

| File | Description |
|---|---|
| `image_depth_pred.png` | Colour-mapped refined depth |
| `image_depth_pred.npy` | Raw refined depth array, shape `[H, W]` |
| `image_normal_pred.png` | Colour-mapped refined surface normals |
| `image_normal_pred.npy` | Raw refined normal array, shape `[H, W, 3]`, range `[-1, 1]` |
| `image_coarse_depth.png` | *(with `--save-intermediates`)* Colour-mapped coarse depth |
| `image_coarse_depth.npy` | *(with `--save-intermediates`)* Raw coarse depth array |
| `image_coarse_normal.png` | *(with `--save-intermediates`)* Colour-mapped coarse normals |
| `image_coarse_normal.npy` | *(with `--save-intermediates`)* Raw coarse normal array |


## **Acknowledgement**


## Citation
If you find our work useful for your research, please consider citing the paper
<!-- ```
@article{li2023patchfusion,
    title={PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation},
    author={Zhenyu Li and Shariq Farooq Bhat and Peter Wonka},
    booktitle={CVPR},
    year={2024}
}
``` -->
