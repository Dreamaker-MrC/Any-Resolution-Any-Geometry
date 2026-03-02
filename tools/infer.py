#!/usr/bin/env python3
"""
URGT: Ultra Resolution Geometry Transformer
Single-image inference script.

Runs the full URGT pipeline on a single input image:
  1. Depth Anything v2  ->  coarse relative depth
  2. Metric3D v2        ->  coarse surface normals
  3. URGT refiner      ->  high-resolution refined depth + normals

Example usage (full pipeline):
    python tools/infer.py \\
        --image    path/to/image.jpg \\
        --checkpoint path/to/urgt_checkpoint.pth \\
        --output-dir ./output

Example usage (with pre-computed coarse predictions):
    python tools/infer.py \\
        --image        path/to/image.jpg \\
        --checkpoint   path/to/urgt_checkpoint.pth \\
        --coarse-depth path/to/coarse_depth.npy \\
        --coarse-normal path/to/coarse_normal.npy \\
        --output-dir   ./output
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

# Resolve project root so the script can be run from any working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Add Metric3D v2 source to path so its internal imports resolve correctly
_METRIC3D_DIR = os.path.join(_PROJECT_ROOT, 'src', 'external', 'metric3d_v2')
if _METRIC3D_DIR not in sys.path:
    sys.path.insert(0, _METRIC3D_DIR)

# Stub out src package and heavy sub-packages to prevent cascade imports
# (e.g. src/datasets imports kornia which may not be installed)
import types as _types

def _register_stub(name):
    """Register a lightweight namespace stub for a package, skipping its __init__.py."""
    if name not in sys.modules:
        mod = _types.ModuleType(name)
        parts = name.split('.')
        mod.__path__ = [os.path.join(_PROJECT_ROOT, *parts)]
        mod.__package__ = name
        sys.modules[name] = mod

for _pkg in ('src', 'src.models', 'src.utils', 'src.external'):
    _register_stub(_pkg)

from src.external.depth_anything_v2.dpt import DepthAnythingV2
from src.models.URGT import URGTModel
from src.utils.color import colorize, colorize_normal

_DAV2_CHECKPOINT_URLS = {
    'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
    'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
    'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
    'vitg': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth',
}
_DAV2_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='URGT single-image inference: estimates high-resolution depth and surface normals.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    req = parser.add_argument_group('required arguments')
    req.add_argument(
        '--image', required=True, metavar='PATH',
        help='Path to the input RGB image (JPG/PNG).',
    )
    req.add_argument(
        '--checkpoint', required=True, metavar='PATH',
        help='Path to the URGT model checkpoint (.pth).',
    )

    out = parser.add_argument_group('output arguments')
    out.add_argument(
        '--output-dir', default=None, metavar='DIR',
        help='Directory to save results. Defaults to the same directory as the input image.',
    )
    out.add_argument(
        '--save-intermediates', action='store_true',
        help='Also save visualisations of the coarse depth and normal maps.',
    )

    dav2 = parser.add_argument_group(
        'Depth Anything v2 (coarse depth)',
        'These options are ignored when --coarse-depth is provided.',
    )
    dav2.add_argument(
        '--dav2-encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
        metavar='ENCODER',
        help='Depth Anything v2 encoder variant. '
             'Larger encoders are more accurate but slower. '
             'Choices: vits (fast) / vitb / vitl (default) / vitg (best quality).',
    )
    dav2.add_argument(
        '--coarse-depth', default=None, metavar='PATH',
        help='Path to a pre-computed coarse depth map (.npy, shape [H, W]). '
             'When provided, Depth Anything v2 is not run.',
    )

    m3d = parser.add_argument_group(
        'Metric3D v2 (coarse surface normals)',
        'These options are ignored when --coarse-normal is provided.',
    )
    m3d.add_argument(
        '--metric3d-model', default='ViT-Small',
        choices=['ViT-Small', 'ViT-Large', 'ViT-giant2'],
        metavar='MODEL',
        help='Metric3D v2 model variant. '
             'Choices: ViT-Small (default, fast) / ViT-Large / ViT-giant2 (best quality).',
    )
    m3d.add_argument(
        '--coarse-normal', default=None, metavar='PATH',
        help='Path to a pre-computed coarse normal map (.npy, shape [H, W, 3], range [-1, 1]). '
             'When provided, Metric3D v2 is not run.',
    )

    patch = parser.add_argument_group('URGT patch configuration')
    patch.add_argument(
        '--patch-split', nargs=2, type=int, default=[8, 8],
        metavar=('N_H', 'N_W'),
        help='Number of patches along height and width (default: 8 8). '
             'The image will be resized so that its dimensions are divisible by these values.',
    )
    patch.add_argument(
        '--min-depth', type=float, default=1e-3,
        help='Minimum depth value in metres (default: 0.001).',
    )
    patch.add_argument(
        '--max-depth', type=float, default=80.0,
        help='Maximum depth value in metres (default: 80).',
    )

    hw = parser.add_argument_group('hardware')
    hw.add_argument(
        '--device', default=None, choices=['cuda', 'cpu'],
        help='Compute device. Defaults to CUDA if available, otherwise CPU.',
    )

    return parser.parse_args()


def build_patch_processor(image_h, image_w, patch_split=(8, 8)):
    """Return a patch-processor namespace and the aligned (H, W) for the given image size."""
    n_h, n_w = patch_split
    aligned_h = (image_h // n_h) * n_h
    aligned_w = (image_w // n_w) * n_w
    ph = aligned_h // n_h
    pw = aligned_w // n_w
    patch_proc = SimpleNamespace(
        patch_split_num=(n_h, n_w),
        coarse_process_shape=(aligned_h, aligned_w),
        patch_process_shape=(518, 518),
        image_shape=(aligned_h, aligned_w),
        patch_shape=(ph, pw),
        raw_h_split_point=[i * ph for i in range(n_h)],
        raw_w_split_point=[i * pw for i in range(n_w)],
    )
    return patch_proc, aligned_h, aligned_w


def load_dav2_model(encoder='vitl', device='cuda'):
    """Load Depth Anything v2. Checkpoint is downloaded automatically."""
    print(f'  Loading Depth Anything v2 ({encoder}) ... ', end='', flush=True)
    model = DepthAnythingV2(**_DAV2_CONFIGS[encoder])
    state_dict = torch.hub.load_state_dict_from_url(
        _DAV2_CHECKPOINT_URLS[encoder], map_location='cpu',
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print('done.')
    return model


def run_dav2(model, image_bgr, input_size=518):
    """Run Depth Anything v2; returns [H, W] float32 depth normalised to [0, 1]."""
    depth = model.infer_image(image_bgr, input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def load_metric3d_model(model_type='ViT-Small', device='cuda'):
    """Load Metric3D v2. Checkpoint is downloaded automatically from HuggingFace."""
    from hubconf import metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2

    _loaders = {
        'ViT-Small':  metric3d_vit_small,
        'ViT-Large':  metric3d_vit_large,
        'ViT-giant2': metric3d_vit_giant2,
    }
    print(f'  Loading Metric3D v2 ({model_type}) ... ', end='', flush=True)
    model = _loaders[model_type](pretrain=True)
    model = model.to(device).eval()
    print('done.')
    return model


def run_metric3d(model, image_rgb, device='cuda'):
    """Run Metric3D v2; returns [H, W, 3] float32 unit surface normals in [-1, 1]."""
    h_orig, w_orig = image_rgb.shape[:2]
    input_size = (616, 1064)  # ViT model input size (H, W)

    scale = min(input_size[0] / h_orig, input_size[1] / w_orig)
    rgb_resized = cv2.resize(
        image_rgb,
        (int(w_orig * scale), int(h_orig * scale)),
        interpolation=cv2.INTER_LINEAR,
    )

    # Pad to exact input size using ImageNet mean as fill colour
    padding_color = [123.675, 116.28, 103.53]
    h_r, w_r = rgb_resized.shape[:2]
    pad_h = input_size[0] - h_r
    pad_w = input_size[1] - w_r
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb_padded = cv2.copyMakeBorder(
        rgb_resized,
        pad_h_half, pad_h - pad_h_half,
        pad_w_half, pad_w - pad_w_half,
        cv2.BORDER_CONSTANT, value=padding_color,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std  = torch.tensor([58.395,  57.12,  57.375]).float()[:, None, None]
    rgb_t = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float()
    rgb_t = torch.div((rgb_t - mean), std).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, output_dict = model.inference({'input': rgb_t})

    pred_normal = output_dict['prediction_normal'][:, :3, :, :].squeeze(0)  # take first 3 channels; 4th is confidence

    h_end = pred_normal.shape[1] - pad_info[1] if pad_info[1] > 0 else pred_normal.shape[1]
    w_end = pred_normal.shape[2] - pad_info[3] if pad_info[3] > 0 else pred_normal.shape[2]
    pred_normal = pred_normal[:, pad_info[0]:h_end, pad_info[2]:w_end]

    pred_normal = F.interpolate(
        pred_normal.unsqueeze(0), size=(h_orig, w_orig),
        mode='bilinear', align_corners=True,
    ).squeeze(0)

    return pred_normal.cpu().numpy().transpose(1, 2, 0)


def build_urgt_model(checkpoint_path, patch_processor, min_depth=1e-3, max_depth=80.0, device='cuda'):
    """Instantiate and load the URGT joint depth-normal refinement model."""
    print('  Loading URGT model ... ', end='', flush=True)
    model = URGTModel(
        min_depth=min_depth,
        max_depth=max_depth,
        patch_processor=patch_processor,
        pretrained_model_path=checkpoint_path,
    )
    model = model.to(device).eval()
    print('done.')
    return model


def main():
    args = parse_args()

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    if not os.path.isfile(args.image):
        sys.exit(f'[Error] Image not found: {args.image}')
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        sys.exit(f'[Error] Failed to read image: {args.image}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = image_bgr.shape[:2]
    print(f'Input image: {args.image}  ({W_orig} x {H_orig})')

    patch_split = tuple(args.patch_split)
    patch_proc, H, W = build_patch_processor(H_orig, W_orig, patch_split)
    if (H, W) != (H_orig, W_orig):
        print(f'[Info] Image resized from {H_orig}x{W_orig} to {H}x{W} '
              f'to align with {patch_split[0]}x{patch_split[1]} patch grid.')
        image_bgr = cv2.resize(image_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.image))
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    if args.coarse_depth:
        print(f'\n[1/3] Loading pre-computed coarse depth from: {args.coarse_depth}')
        coarse_depth = np.load(args.coarse_depth).astype(np.float32)
        d_max = coarse_depth.max()
        coarse_depth = coarse_depth / (d_max + 1e-8)  # normalise to [0, 1]
    else:
        print(f'\n[1/3] Running Depth Anything v2 (encoder: {args.dav2_encoder}) ...')
        dav2_model = load_dav2_model(args.dav2_encoder, device)
        coarse_depth = run_dav2(dav2_model, image_bgr)
        del dav2_model
        if device == 'cuda':
            torch.cuda.empty_cache()

    if coarse_depth.shape != (H, W):
        coarse_depth = cv2.resize(coarse_depth, (W, H), interpolation=cv2.INTER_LINEAR)
    print(f'  Coarse depth ready. shape={coarse_depth.shape}, '
          f'range=[{coarse_depth.min():.3f}, {coarse_depth.max():.3f}]')

    if args.save_intermediates:
        depth_coarse_vis = colorize(
            torch.from_numpy(coarse_depth).unsqueeze(0).unsqueeze(0),
            cmap='turbo', vminp=0, vmaxp=100,
        )
        cv2.imwrite(os.path.join(args.output_dir, f'{stem}_coarse_depth.png'), depth_coarse_vis)
        np.save(os.path.join(args.output_dir, f'{stem}_coarse_depth.npy'), coarse_depth)

    if args.coarse_normal:
        print(f'\n[2/3] Loading pre-computed coarse normals from: {args.coarse_normal}')
        coarse_normal = np.load(args.coarse_normal).astype(np.float32)
    else:
        print(f'\n[2/3] Running Metric3D v2 (model: {args.metric3d_model}) ...')
        metric3d_model = load_metric3d_model(args.metric3d_model, device)
        coarse_normal = run_metric3d(metric3d_model, image_rgb, device)
        del metric3d_model
        if device == 'cuda':
            torch.cuda.empty_cache()

    if coarse_normal.shape[:2] != (H, W):
        coarse_normal = cv2.resize(coarse_normal, (W, H), interpolation=cv2.INTER_LINEAR)
    print(f'  Coarse normal ready. shape={coarse_normal.shape}, '
          f'range=[{coarse_normal.min():.3f}, {coarse_normal.max():.3f}]')

    if args.save_intermediates:
        normal_np = coarse_normal.transpose(2, 0, 1)
        coarse_normal_vis = colorize_normal(
            torch.from_numpy(normal_np[[2, 1, 0]]),  # channel swap for vis
        )
        cv2.imwrite(os.path.join(args.output_dir, f'{stem}_coarse_normal.png'), coarse_normal_vis)
        np.save(os.path.join(args.output_dir, f'{stem}_coarse_normal.npy'), coarse_normal)

    print('\n[3/3] Running URGT refinement ...')
    urgt_model = build_urgt_model(
        args.checkpoint, patch_proc,
        min_depth=args.min_depth, max_depth=args.max_depth,
        device=device,
    )

    image_t = (
        torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # model adds the channel dim internally via unsqueeze
    depth_t = torch.from_numpy(coarse_depth).float().unsqueeze(0).to(device)

    normal_np = coarse_normal.transpose(2, 0, 1).astype(np.float32)
    norm_mag = np.linalg.norm(normal_np, axis=0, keepdims=True)
    normal_np = normal_np / (norm_mag + 1e-12)
    normal_t = torch.from_numpy(normal_np).unsqueeze(0).to(device)

    with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=torch.bfloat16):
        with torch.no_grad():
            _, log_dict = urgt_model(
                image_highres=image_t.half(),
                coarse_depth=depth_t.half(),
                coarse_normal=normal_t.half(),
                depth_valid_mask=torch.ones_like(depth_t).half(),
                normal_valid_mask=torch.ones_like(normal_t).half(),
            )

    depth_pred  = log_dict['depth_pred']
    normal_pred = log_dict['normal_pred'].squeeze(0)[[2, 1, 0]]

    normal_pred = F.normalize(normal_pred.float(), dim=0)

    np.save(
        os.path.join(args.output_dir, f'{stem}_depth_pred.npy'),
        depth_pred.squeeze().cpu().float().numpy(),
    )
    np.save(
        os.path.join(args.output_dir, f'{stem}_normal_pred.npy'),
        normal_pred.permute(1, 2, 0).cpu().float().numpy(),
    )

    depth_vis  = colorize(depth_pred, cmap='turbo', vminp=0, vmaxp=100)
    normal_vis = colorize_normal(normal_pred)
    cv2.imwrite(os.path.join(args.output_dir, f'{stem}_depth_pred.png'),  depth_vis)
    cv2.imwrite(os.path.join(args.output_dir, f'{stem}_normal_pred.png'), normal_vis)

    depth_np = depth_pred.squeeze().cpu().float().numpy()
    normal_np_out = normal_pred.cpu().float().numpy()
    print(f'\nDone. Results saved to: {args.output_dir}')
    print(f'  depth  -> shape={depth_np.shape}, range=[{depth_np.min():.3f}, {depth_np.max():.3f}] (relative)')
    print(f'  normal -> shape={normal_np_out.shape}, range=[{normal_np_out.min():.3f}, {normal_np_out.max():.3f}]')


if __name__ == '__main__':
    main()
