import matplotlib
import matplotlib.cm
import numpy as np
import torch

def colorize(
        value,
        vmin=None,
        vmax=None,
        cmap='turbo_r',
        invalid_val=-99,
        invalid_mask=None,
        background_color=(128, 128, 128, 255),
        gamma_corrected=False,
        value_transform=None,
        vminp=2,
        vmaxp=95):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarray): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W).
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap.
        vmax (float, optional): vmax-valued entries are mapped to end color of cmap.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'turbo_r'.
        invalid_val (int, optional): Value of invalid pixels, colored as background_color. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions.
        background_color (tuple[int], optional): 4-tuple RGB color for invalid pixels.
        gamma_corrected (bool, optional): Apply gamma correction to colored image.
        value_transform (Callable, optional): Transform applied to valid pixels before coloring.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = value.squeeze()

    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    if invalid_mask is None:
        invalid_mask = value == invalid_val

    if isinstance(invalid_mask, torch.Tensor):
        invalid_mask = invalid_mask.detach().cpu().numpy()
    invalid_mask = invalid_mask.squeeze()

    mask = np.logical_not(invalid_mask)

    vmin = np.percentile(value[mask],vminp) if vmin is None else vmin
    vmax = np.percentile(value[mask],vmaxp) if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[...]
    img[invalid_mask] = background_color

    if gamma_corrected:
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def colorize_normal(
    normal,
    valid_mask=None,
    invalid_mask=None,
    invalid_val=None,
    background_color=(255, 255, 255),
    gamma_corrected=False
):
    """
    Visualize a normal map as a color image.
    normal: (C,H,W) or (B,C,H,W), range (-1,1). Returns np.ndarray (H,W,3) uint8.
    """
    if isinstance(normal, torch.Tensor):
        normal = normal.detach().cpu().numpy()
    if normal.ndim == 4:
        normal = normal[0]
    if normal.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {normal.shape[0]}")

    norm = np.linalg.norm(normal, axis=0, keepdims=True)
    normal = normal / (norm + 1e-6)

    normal = normal.transpose(1, 2, 0)

    if invalid_mask is None:
        if invalid_val is not None:
            invalid_mask = np.any(np.abs(normal - invalid_val) < 1e-6, axis=-1)
        else:
            invalid_mask = np.all(np.abs(normal) < 1e-6, axis=-1)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.detach().cpu().numpy().squeeze()
        invalid_mask = ~valid_mask.astype(bool)

    if isinstance(invalid_mask, torch.Tensor):
        invalid_mask = invalid_mask.detach().cpu().numpy().squeeze()

    normal_vis = (normal + 1.0) / 2.0
    normal_vis = np.clip(normal_vis, 0.0, 1.0)

    img = (normal_vis * 255).astype(np.uint8)

    if gamma_corrected:
        img = img.astype(np.float32) / 255.0
        img = np.power(img, 1/2.2)
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    return img
