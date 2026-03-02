# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from src.external.vggt.models.aggregator_joint import Aggregator_Joint
from src.external.vggt.heads.camera_head import CameraHead
from src.external.vggt.heads.dpt_head import DPTHead
from src.external.vggt.heads.track_head import TrackHead


class VGGT_Joint(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator_Joint(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="no_constrain", conf_activation="expp1")
        self.normal_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="no_constrain", conf_activation="expp1")

        self.camera_head = None
        self.point_head = None
        self.track_head = None

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        depth_prior: torch.Tensor = None,
        normal_prior: torch.Tensor = None,
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images, depth_prior, normal_prior)

        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.normal_head is not None:
                normal, normal_conf = self.normal_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["normal"] = normal
                predictions["normal_conf"] = normal_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
