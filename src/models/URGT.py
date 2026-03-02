# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Wenqing Cui

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from src.models.utils import SimpleResizer

from torchvision.ops import roi_align as torch_roi_align
from src.external.vggt.models.vggt_joint import VGGT_Joint
from src.registry import MODELS

from huggingface_hub import PyTorchModelHubMixin

@MODELS.register_module()
class URGTModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 min_depth,
                 max_depth,
                 patch_processor,
                 model_infer_cfg=None,
                 pretrained_model_path=None,
                 **kwargs):
        super().__init__()

        self.patch_processor = patch_processor
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pretrained_model_path = pretrained_model_path
        self.model_infer_cfg = model_infer_cfg

        self.vggt = VGGT_Joint()

        if self.pretrained_model_path is not None:
            print(f"Loading pretrained model from {self.pretrained_model_path}")
            print_log(self.vggt.load_state_dict(torch.load(self.pretrained_model_path, weights_only=False)['model_state_dict'], strict=False), logger='current')
        else:
            print_log(self.vggt.load_state_dict(torch.load('./work_dir/ckpts/vggt_1B.pt', weights_only=False), strict=False), logger='current')
            nn.init.normal_(self.vggt.depth_head.scratch.output_conv2[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.vggt.depth_head.scratch.output_conv2[-1].bias)
        self.vggt.track_head = None
        self.vggt.point_head = None

        self.coarse_resizer = SimpleResizer(self.patch_processor.coarse_process_shape[0], self.patch_processor.coarse_process_shape[1])
        self.patch_resizer = SimpleResizer(self.patch_processor.patch_process_shape[0], self.patch_processor.patch_process_shape[1])

    def refiner_model_forward(self, patch_image, coarse_depth_prediction_roi, coarse_normal_prediction_roi):

        pred_dict = self.vggt(patch_image, depth_prior=coarse_depth_prediction_roi, normal_prior=coarse_normal_prediction_roi)

        depth_offset_prediction = pred_dict['depth']
        depth_pred_confidence = pred_dict['depth_conf']
        normal_offset_prediction = pred_dict['normal']
        normal_confidence = pred_dict['normal_conf']

        coarse_depth_prediction_roi = coarse_depth_prediction_roi.squeeze()
        depth_offset_prediction = depth_offset_prediction.squeeze()
        normal_offset_prediction = normal_offset_prediction.squeeze(dim=0)
        normal_offset_prediction = normal_offset_prediction.permute(0, 3, 1, 2)

        if coarse_depth_prediction_roi.dim() == 2:
            coarse_depth_prediction_roi = coarse_depth_prediction_roi.unsqueeze(dim=0)
        if depth_offset_prediction.dim() == 2:
            depth_offset_prediction = depth_offset_prediction.unsqueeze(dim=0)

        resize_shape = self.patch_processor.patch_shape

        coarse_depth_prediction_roi = F.interpolate(
            coarse_depth_prediction_roi.unsqueeze(dim=0), size=resize_shape, mode='bilinear', align_corners=True)
        depth_offset_prediction = F.interpolate(
            depth_offset_prediction.unsqueeze(dim=0), size=resize_shape, mode='bilinear', align_corners=True)
        coarse_normal_prediction_roi = F.interpolate(
            coarse_normal_prediction_roi, size=resize_shape, mode='bilinear', align_corners=True)
        normal_offset_prediction = F.interpolate(
            normal_offset_prediction, size=resize_shape, mode='bilinear', align_corners=True)

        depth_refined_prediction = coarse_depth_prediction_roi + depth_offset_prediction
        normal_refined_prediction = coarse_normal_prediction_roi + normal_offset_prediction
        log_dict = {
            'depth_conf': depth_pred_confidence,
            'depth_pred': depth_refined_prediction,
            'normal_pred': normal_refined_prediction,
            'normal_conf': normal_confidence,
            'depth_offset_prediction': depth_offset_prediction,
            'normal_offset_prediction': normal_offset_prediction
        }
        return log_dict

    def bbox_process(self, bboxs):
        bs, n_rois, _ = bboxs.shape
        bboxs_feat_factor = torch.tensor([
                1 / self.patch_processor.image_shape[1] * self.patch_processor.coarse_process_shape[1],
                1 / self.patch_processor.image_shape[0] * self.patch_processor.coarse_process_shape[0],
                1 / self.patch_processor.image_shape[1] * self.patch_processor.coarse_process_shape[1],
                1 / self.patch_processor.image_shape[0] * self.patch_processor.coarse_process_shape[0]], device=bboxs.device).unsqueeze(dim=0)

        bboxs_feat = bboxs * bboxs_feat_factor

        inds = torch.arange(bs).to(bboxs.device).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, n_rois, 1)
        bboxs_feat = torch.cat((inds, bboxs_feat), dim=-1)
        bboxs_feat = bboxs_feat.view(bs * n_rois, 5)
        return bboxs_feat

    def forward(
        self,
        image_highres=None,
        coarse_depth=None,
        coarse_normal=None,
        depth_valid_mask=None,
        normal_valid_mask=None,
        **kwargs):

        if self.model_infer_cfg is not None:
            USE_OVERLAP = self.model_infer_cfg['overlap_infer_mode']
        else:
            USE_OVERLAP = True

        if USE_OVERLAP:

            patch_h, patch_w = self.patch_processor.patch_shape
            img_h, img_w = self.patch_processor.image_shape

            overlap_h = int(patch_h * 0.3)
            overlap_w = int(patch_w * 0.3)

            stride_h = patch_h - overlap_h
            stride_w = patch_w - overlap_w
            h_splits = list(range(0, img_h - patch_h + 1, stride_h))
            w_splits = list(range(0, img_w - patch_w + 1, stride_w))

            if h_splits[-1] + patch_h < img_h:
                h_splits.append(img_h - patch_h)
            if w_splits[-1] + patch_w < img_w:
                w_splits.append(img_w - patch_w)

            imgs_crop = []
            bboxs = []
            for h_start in h_splits:
                for w_start in w_splits:
                    crop_image = image_highres[:, :, h_start: h_start + patch_h, w_start: w_start + patch_w]
                    crop_image_resized = self.patch_resizer(crop_image)
                    bbox = torch.tensor([w_start, h_start, w_start + patch_w, h_start + patch_h])
                    imgs_crop.append(crop_image_resized)
                    bboxs.append(bbox)

            imgs_crop = torch.stack(imgs_crop, dim=0)
            bboxs = torch.stack(bboxs, dim=0)

            imgs_crop = imgs_crop.to(image_highres.device)
            imgs_crop = imgs_crop.transpose(0, 1)  # NOTE: transpose to align batch and sequence dimensions

            bboxs = bboxs.to(image_highres.device).int().unsqueeze(dim=0)
            bs, n_rois, _ = bboxs.shape
            bboxs_feat = self.bbox_process(bboxs)
            coarse_depth_prediction = coarse_depth.unsqueeze(dim=0)
            coarse_normal_prediction = coarse_normal
            depth_coarse_prediction_roi = torch_roi_align(
                coarse_depth_prediction,
                bboxs_feat,
                self.patch_processor.patch_process_shape,
                coarse_depth_prediction.shape[-2] / self.patch_processor.coarse_process_shape[0],
                aligned=True
            )
            _, _, h, w = depth_coarse_prediction_roi.shape

            depth_coarse_prediction_roi = depth_coarse_prediction_roi.view(bs, n_rois, 1, h, w)

            normal_coarse_prediction_roi = torch_roi_align(
                coarse_normal_prediction,
                bboxs_feat,
                self.patch_processor.patch_process_shape,
                coarse_normal_prediction.shape[-2] / self.patch_processor.coarse_process_shape[0],
                aligned=True
            )
            _, _, h, w = normal_coarse_prediction_roi.shape

            normal_coarse_prediction_roi = normal_coarse_prediction_roi.view(bs, n_rois, 3, h, w)

            log_dict = self.refiner_model_forward(imgs_crop, depth_coarse_prediction_roi, normal_coarse_prediction_roi.squeeze(0))
            depth_refined_prediction = log_dict['depth_pred'].squeeze()
            depth_offset_prediction = log_dict['depth_offset_prediction'].squeeze()
            normal_offset_prediction = log_dict['normal_offset_prediction'].squeeze()
            normal_refined_prediction = log_dict['normal_pred'].squeeze(dim=0)

            device = depth_refined_prediction.device
            dtype = depth_refined_prediction.dtype

            # 2D Gaussian blend weights (attenuates patch edges during overlap averaging)
            sigma_h = patch_h * 0.4
            sigma_w = patch_w * 0.4
            coords_h = (torch.arange(patch_h, device=device, dtype=dtype) - (patch_h - 1) / 2.0)
            coords_w = (torch.arange(patch_w, device=device, dtype=dtype) - (patch_w - 1) / 2.0)
            gauss_h = torch.exp(-0.5 * (coords_h / sigma_h) ** 2)
            gauss_w = torch.exp(-0.5 * (coords_w / sigma_w) ** 2)
            blend_mask = (gauss_h[:, None] * gauss_w[None, :])
            blend_mask = blend_mask / (blend_mask.max() + 1e-12)

            blend_mask = torch.ones((patch_h, patch_w), device=device, dtype=dtype)

            pred_depth_accum = torch.zeros((img_h, img_w), device=device, dtype=dtype)
            weight_accum = torch.zeros((img_h, img_w), device=device, dtype=dtype)
            pred_normal_accum = torch.zeros((3, img_h, img_w), device=device, dtype=dtype)
            patch_select_idx = 0
            for h_start in h_splits:
                for w_start in w_splits:
                    temp_depth = depth_refined_prediction[patch_select_idx]
                    temp_normal = normal_refined_prediction[patch_select_idx]
                    h_end = h_start + patch_h
                    w_end = w_start + patch_w

                    pred_depth_accum[h_start:h_end, w_start:w_end] += temp_depth * blend_mask
                    weight_accum[h_start:h_end, w_start:w_end] += blend_mask
                    pred_normal_accum[:, h_start:h_end, w_start:w_end] += temp_normal * blend_mask
                    patch_select_idx += 1

            eps = 1e-8
            pred_depth = (pred_depth_accum / (weight_accum + eps)).unsqueeze(0).unsqueeze(0)
            pred_normal = (pred_normal_accum / (weight_accum + eps)).unsqueeze(0)
            log_dict = {
                'depth_pred': pred_depth,
                'normal_pred': pred_normal,
            }

        else:
            height, width = self.patch_processor.patch_shape[0], self.patch_processor.patch_shape[1]

            imgs_crop = []
            bboxs = []

            for h_start in self.patch_processor.raw_h_split_point:
                for w_start in self.patch_processor.raw_w_split_point:
                    crop_image = image_highres[:, :, h_start: h_start+height, w_start: w_start+width]
                    crop_image_resized = self.patch_resizer(crop_image)
                    bbox = torch.tensor([w_start, h_start, w_start+width, h_start+height])
                    imgs_crop.append(crop_image_resized)
                    bboxs.append(bbox)

            imgs_crop = torch.stack(imgs_crop, dim=0)
            bboxs = torch.stack(bboxs, dim=0)

            imgs_crop = imgs_crop.to(image_highres.device)
            imgs_crop = imgs_crop.transpose(0, 1)  # NOTE: transpose to align batch and sequence dimensions

            bboxs = bboxs.to(image_highres.device).int().unsqueeze(dim=0)
            bs, n_rois, _ = bboxs.shape
            bboxs_feat = self.bbox_process(bboxs)

            coarse_depth_prediction = coarse_depth.unsqueeze(dim=0)
            coarse_normal_prediction = coarse_normal.unsqueeze(dim=0)

            depth_coarse_prediction_roi = torch_roi_align(
                coarse_depth_prediction,
                bboxs_feat,
                self.patch_processor.patch_process_shape,
                coarse_depth_prediction.shape[-2] / self.patch_processor.coarse_process_shape[0],
                aligned=True)
            _, _, h, w = depth_coarse_prediction_roi.shape
            depth_coarse_prediction_roi = depth_coarse_prediction_roi.view(bs, n_rois, 1, h, w)

            normal_coarse_prediction_roi = torch_roi_align(
                coarse_normal_prediction.squeeze(dim=0),
                bboxs_feat,
                self.patch_processor.patch_process_shape,
                coarse_normal_prediction.shape[-2] / self.patch_processor.coarse_process_shape[0],
                aligned=True
            )
            _, _, h, w = normal_coarse_prediction_roi.shape
            normal_coarse_prediction_roi = normal_coarse_prediction_roi.view(bs, n_rois, 3, h, w)

            log_dict = self.refiner_model_forward(imgs_crop, depth_coarse_prediction_roi, normal_coarse_prediction_roi.squeeze(0))
            depth_refined_prediction = log_dict['depth_pred'].squeeze()
            normal_refined_prediction = log_dict['normal_pred'].squeeze(dim=0)

            pred_depth = torch.zeros(self.patch_processor.image_shape, device=depth_refined_prediction.device)
            pred_normal = torch.zeros((3,) + self.patch_processor.image_shape, device=normal_refined_prediction.device)
            patch_select_idx = 0
            for h_start in self.patch_processor.raw_h_split_point:
                for w_start in self.patch_processor.raw_w_split_point:
                    temp_depth = depth_refined_prediction[patch_select_idx]
                    temp_normal = normal_refined_prediction[patch_select_idx]
                    pred_depth[h_start: h_start+height, w_start: w_start+width] = temp_depth
                    pred_normal[:, h_start: h_start+height, w_start: w_start+width] = temp_normal
                    patch_select_idx += 1

            pred_depth = pred_depth.unsqueeze(dim=0).unsqueeze(dim=0)
            pred_normal = pred_normal.unsqueeze(dim=0)
            log_dict = {
                'depth_pred': pred_depth,
                'normal_pred': pred_normal,
            }

        return pred_depth, log_dict
