from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform

from .loss import contour_loss_func, contour_raw_loss_func
import cv2

BASIS_MODULE_REGISTRY = Registry("BASIS_MODULE")
BASIS_MODULE_REGISTRY.__doc__ = """
Registry for basis module, which produces global bases from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


def build_basis_module(cfg, input_shape):
    name = cfg.MODEL.BASIS_MODULE.NAME
    return BASIS_MODULE_REGISTRY.get(name)(cfg, input_shape)


@BASIS_MODULE_REGISTRY.register()
class ProtoNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.BASIS_MODULE.NUM_BASES
        planes            = cfg.MODEL.BASIS_MODULE.CONVS_DIM
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES
        self.loss_on      = cfg.MODEL.BASIS_MODULE.LOSS_ON
        norm              = cfg.MODEL.BASIS_MODULE.NORM
        num_convs         = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize    = cfg.MODEL.SEPSNET.VISUALIZE
        self.dilation_on  = cfg.MODEL.BASIS_MODULE.DILATION
        self.contour_on   = cfg.MODEL.BASIS_MODULE.BASIS_CONTOUR_ON
        self.one_contour  = cfg.MODEL.BASIS_MODULE.ONE_BASIS_CONTOUR_ON
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))
        tower = []
        for i in range(num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower', nn.Sequential(*tower))
        
        if self.dilation_on:
            tower = []
            for i in range(num_convs):
                tower.append(
                    conv_block(planes, planes, 3, 1, 3))
            tower.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            tower.append(
                conv_block(planes, planes, 3, 1, 3))
            tower.append(
                nn.Conv2d(planes, mask_dim, 1))
            self.add_module('tower_dilation', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(conv_block(inplanes, planes, 3, 1),
                                          conv_block(planes, planes, 3, 1),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))

    def forward(self, features, targets=None, targets_contour=None, gt_instances=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=False)
                # x_p = aligned_bilinear(x_p, x.size(3) // x_p.size(3))
                x = x + x_p
        outputs = {"bases": [self.tower(x)]}
        if self.dilation_on:
            outputs.update({"bases_dilation": [self.tower_dilation(x)]})
        losses = {}
        
        # auxiliary thing semantic loss
        if self.training and self.loss_on:
            sem_out = self.seg_head(features[self.in_features[0]])
            # resize target to reduce memory
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.common_stride)
            seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight
            
        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
            
        if self.training and self.contour_on:
            gt_sem_onehot = torch.zeros_like(sem_out).to(device=sem_out.device)
            gt_sem_onehot = gt_sem_onehot.scatter_(1, gt_sem.long(), 1)
            N, C, H, W = sem_out.shape
            contour_loss = contour_loss_func(sem_out.view(-1, H, W), gt_sem_onehot.view(-1, H, W)).mean()
            losses['loss_basis_cont'] = contour_loss * 0.5
                
        if self.training and self.one_contour:
            contour_basis = outputs["bases"][0][:, -1, :]
            contour_loss = contour_raw_loss_func(contour_basis, targets_contour).mean()
            losses['loss_one_basis_cont'] = contour_loss

        
            
        return outputs, losses
