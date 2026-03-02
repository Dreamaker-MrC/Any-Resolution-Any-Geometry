import torch

def split_global_pos(global_pos, num_patch_h, num_patch_w, patch_h, patch_w):
    """
    Split global raster-scan positions into patch-wise positions.

    Args:
        global_pos: Tensor of shape [1, H*W, 2], raster-scan order (row-major).
        num_patch_h: Number of patches along height (e.g., 4).
        num_patch_w: Number of patches along width (e.g., 4).
        patch_h: Tokens per patch along height (e.g., 37).
        patch_w: Tokens per patch along width (e.g., 37).

    Returns:
        patch_pos: Tensor of shape [num_patch_h*num_patch_w, patch_h*patch_w, 2].
    """
    H, W = num_patch_h * patch_h, num_patch_w * patch_w

    # reshape to [H, W, 2]
    
    pos_hw = global_pos.view(H, W, 2)

    patches = []
    for i in range(num_patch_h):
        for j in range(num_patch_w):
            patch = pos_hw[
                i*patch_h:(i+1)*patch_h,
                j*patch_w:(j+1)*patch_w,
                :
            ]  # shape [patch_h, patch_w, 2]
            patches.append(patch.reshape(-1, 2))  # flatten to [patch_h*patch_w, 2]

    patch_pos = torch.stack(patches, dim=0)  # [num_patch, patch_h*patch_w, 2]
    return patch_pos
