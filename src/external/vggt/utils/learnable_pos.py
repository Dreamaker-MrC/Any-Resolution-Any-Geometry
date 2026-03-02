import torch
import torch.nn as nn

class LearnableGlobalPositionalEncoding(nn.Module):
    def __init__(self, num_patch: int, num_token: int, dim: int):
        super().__init__()
        patch_per_side = int(num_patch ** 0.5)
        tokens_per_patch_side = int(num_token ** 0.5)
        global_size = patch_per_side * tokens_per_patch_side

        # 可学习的参数矩阵 (148 x 148 x dim)
        self.pos_embed = nn.Parameter(torch.zeros(global_size, global_size, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 常见初始化方式

        self.num_patch = num_patch
        self.num_token = num_token
        self.dim = dim
        self.patch_per_side = patch_per_side
        self.tokens_per_patch_side = tokens_per_patch_side

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: Tensor [num_patch, num_token, dim]
        Returns:
            Tensor with learnable positional embedding added.
        """
        encoded_patches = []
        for py in range(self.patch_per_side):
            for px in range(self.patch_per_side):
                y_start, y_end = py*self.tokens_per_patch_side, (py+1)*self.tokens_per_patch_side
                x_start, x_end = px*self.tokens_per_patch_side, (px+1)*self.tokens_per_patch_side
                pos_patch = self.pos_embed[y_start:y_end, x_start:x_end, :]  # [tokens_per_patch_side, tokens_per_patch_side, dim]
                pos_patch = pos_patch.reshape(self.num_token, self.dim)
                patch_id = py*self.patch_per_side + px
                encoded_patches.append(patch_tokens[patch_id] + pos_patch)

        return torch.stack(encoded_patches, dim=0)  # [num_patch, num_token, dim]
