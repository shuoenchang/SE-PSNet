import torch
from torch.nn import functional as F
from torch import nn

from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler
from adet.layers import conv_with_kaiming_uniform, l2_loss

from .loss import contour_loss_func, iou_calc, iou_loss_func, contour_calc


def build_blender(cfg):
    return Blender(cfg)


class Blender(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # fmt: off
        self.pooler_resolution = cfg.MODEL.SEPSNET.BOTTOM_RESOLUTION
        sampling_ratio         = cfg.MODEL.SEPSNET.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.SEPSNET.POOLER_TYPE
        pooler_scales          = cfg.MODEL.SEPSNET.POOLER_SCALES
        self.attn_size         = cfg.MODEL.SEPSNET.ATTN_SIZE
        self.top_interp        = cfg.MODEL.SEPSNET.TOP_INTERP
        self.contour_loss_on   = cfg.MODEL.SEPSNET.CONTOUR_LOSS_ON
        self.iou_loss_on       = cfg.MODEL.SEPSNET.IOU_LOSS_ON
        self.iou_predict_on    = cfg.MODEL.SEPSNET.IOU_PREDICT
        self.contour_weight    = cfg.MODEL.SEPSNET.CONTOUR_WEIGHT
        self.contour_resize    = cfg.MODEL.SEPSNET.CONTOUR_RESIZE
        self.contour_predict   = cfg.MODEL.SEPSNET.CONTOUR_PREDICT
        self.contour_auxiliary = cfg.MODEL.SEPSNET.CONTOUR_AUXILIARY
        self.class_conf_weight = cfg.MODEL.SEPSNET.CLASS_CONFIDENCE_WEIGHT
        self.visualize         = cfg.MODEL.SEPSNET.VISUALIZE
        num_bases              = cfg.MODEL.BASIS_MODULE.NUM_BASES
        # fmt: on

        self.attn_len = num_bases * self.attn_size * self.attn_size

        self.pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=2)

        self.fpn_pooler = ROIPooler(
            output_size=self.pooler_resolution//4,
            scales=(1/8,),
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=3)

        if self.iou_predict_on:
            self.iou_head = IOUPredict(cfg)
        
        if self.contour_auxiliary:
            self.contour_aux_head = AuxConv(cfg)

    def forward(self, bases, proposals, gt_instances, fpn_features):
        if gt_instances is not None:
            # training
            # reshape attns
            losses = {}
            dense_info = proposals["instances"]
            attns = dense_info.top_feats
            pos_inds = dense_info.pos_inds
            if pos_inds.numel() == 0:
                return None, {"loss_mask": sum([x.sum() * 0 for x in attns]) + bases[0].sum() * 0}

            gt_inds = dense_info.gt_inds

            # gen targets
            gt_masks = []
            for instances_per_image in gt_instances:
                if len(instances_per_image.gt_boxes.tensor) == 0:
                    continue
                gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.gt_boxes.tensor, self.pooler_resolution
                ).to(device=attns.device)
                gt_masks.append(gt_mask_per_image)
            gt_masks = cat(gt_masks, dim=0)
            gt_masks = gt_masks[gt_inds]
            N = gt_masks.size(0)
            gt_masks = gt_masks.view(N, -1)

            gt_ctr = dense_info.gt_ctrs
            loss_denorm = proposals["loss_denorm"]

            # merge bases
            bases_rois = self.pooler(bases, [x.gt_boxes for x in gt_instances])
            bases_rois = bases_rois[gt_inds]
            pred_mask_logits = self.merge_bases(bases_rois, attns)
            
            mask_losses = F.binary_cross_entropy_with_logits(
                pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="none")
            mask_loss = ((mask_losses.mean(dim=-1) * gt_ctr).sum()
                         / loss_denorm)
            losses["loss_mask"] = mask_loss

            if self.contour_auxiliary:
                contout_aux_predict = self.contour_aux_head(pred_mask_logits.view(N, 1, self.pooler_resolution, self.pooler_resolution))
                contour_aux_loss = contour_loss_func(contout_aux_predict.view(N, self.pooler_resolution, self.pooler_resolution),
                                            gt_masks.view(N, self.pooler_resolution, self.pooler_resolution), self.contour_resize, False)
                contour_aux_loss = ((contour_aux_loss * gt_ctr).sum()
                            / loss_denorm)
                losses["loss_aux_contour"] = contour_aux_loss * 0.3

            if self.contour_loss_on:
                contour_loss = contour_loss_func(pred_mask_logits.view(N, self.pooler_resolution, self.pooler_resolution),
                                            gt_masks.view(N, self.pooler_resolution, self.pooler_resolution), self.contour_resize)
                contour_loss = ((contour_loss * gt_ctr).sum()
                            / loss_denorm)
                losses["loss_contour"] = contour_loss * 0.5 * self.contour_weight

            if self.iou_loss_on:
                iou_loss = iou_loss_func(pred_mask_logits.view(N, self.pooler_resolution, self.pooler_resolution),
                            gt_masks.view(N, self.pooler_resolution, self.pooler_resolution))
                iou_loss = ((iou_loss * gt_ctr).sum()
                            / loss_denorm)
                losses["loss_iou"] = iou_loss * self.contour_weight

            if self.iou_predict_on:
                gt_iou = iou_calc(pred_mask_logits.view(-1, self.pooler_resolution, self.pooler_resolution),
                            gt_masks.view(-1, self.pooler_resolution, self.pooler_resolution))
                gt_iou = gt_iou.detach()

                fpn_rois = self.fpn_pooler([fpn_features], [x.gt_boxes for x in gt_instances]) # (14, 14)
                fpn_rois = fpn_rois[gt_inds]

                bases_rois_small = F.interpolate(bases_rois, scale_factor=0.25) # (56, 56) -> (14, 14)
                pred_mask_logits_small = F.interpolate(pred_mask_logits.view(N, -1, self.pooler_resolution, self.pooler_resolution), 
                                                    scale_factor=0.25)
                
                iou_head_input = torch.cat((fpn_rois, bases_rois_small, pred_mask_logits_small), 1)
                iou_head_input = iou_head_input.detach() # (N, 261, 14, 14)

                predicts = self.iou_head(iou_head_input)
                predict_iou = predicts[:, 0].sigmoid()
                losses["loss_iou_predict"] = l2_loss(predict_iou, gt_iou)

                if self.contour_predict:
                    contour_score = contour_calc(pred_mask_logits.view(N, self.pooler_resolution, self.pooler_resolution),
                                        gt_masks.view(N, self.pooler_resolution, self.pooler_resolution)).detach()
                    predict_contour = predicts[:, 1].sigmoid()
                    losses["loss_contour_predict"] = l2_loss(predict_contour, contour_score)

            return None, losses
        
        else:
            # no proposals
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                # add empty pred_masks results
                for box in proposals:
                    box.pred_masks = box.pred_classes.view(
                        -1, 1, self.pooler_resolution, self.pooler_resolution)
                return proposals, {}
            N = total_instances
            bases_rois = self.pooler(bases, [x.pred_boxes for x in proposals])
            attns = cat([x.top_feat for x in proposals], dim=0)
            pred_mask_logits = self.merge_bases(bases_rois, attns)
            
            pred_mask_logits = pred_mask_logits.sigmoid()
            pred_mask_logits = pred_mask_logits.view(
                -1, 1, self.pooler_resolution, self.pooler_resolution)

            if self.iou_predict_on:
                fpn_rois = self.fpn_pooler([fpn_features], [x.pred_boxes for x in proposals]) # (14, 14)

                bases_rois_small = F.interpolate(bases_rois, scale_factor=0.25)
                pred_mask_logits_small = F.interpolate(pred_mask_logits.view(N, -1, self.pooler_resolution, self.pooler_resolution), 
                                                    scale_factor=0.25)

                iou_head_input = torch.cat((fpn_rois, bases_rois_small, pred_mask_logits_small), 1)
                predict_iou = self.iou_head(iou_head_input)[:, 0]
                predict_iou = predict_iou.sigmoid()
                if self.contour_predict:
                    predict_contour = self.iou_head(iou_head_input)[:, 1]
                    predict_contour = predict_contour.sigmoid()
                    predict_contour = torch.sqrt(predict_contour)
                    predict_contour = predict_contour
                else:
                    predict_contour = predict_iou

            start_ind = 0
            for box in proposals:
                end_ind = start_ind + len(box)
                box.pred_masks = pred_mask_logits[start_ind:end_ind]
                if self.iou_predict_on:
                    box.scores = box.scores * self.class_conf_weight \
                               + predict_iou[start_ind:end_ind] * (1-self.class_conf_weight) 
                    box.iou_scores = predict_iou[start_ind:end_ind]
                    box.cont_scores = predict_contour[start_ind:end_ind]
                start_ind = end_ind
            
            return proposals, {}

    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()

        coeffs = coeffs.view(N, -1, self.attn_size, self.attn_size)
        coeffs = F.interpolate(coeffs, (H, W),
                               mode=self.top_interp).softmax(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds.view(N, -1)


class IOUPredict(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pooler_resolution = cfg.MODEL.SEPSNET.BOTTOM_RESOLUTION
        num_bases         = cfg.MODEL.BASIS_MODULE.NUM_BASES
        fpn_out_channels  = cfg.MODEL.FPN.OUT_CHANNELS
        conv_block = conv_with_kaiming_uniform(None, True)  # conv relu bn
        in_channels = fpn_out_channels+num_bases+1

        tower = []
        tower.append(conv_block(in_channels, 256, 3, 1))
        tower.append(conv_block(256, 128, 3, 1))
        tower.append(conv_block(128, 32, 3, 1))
        tower.append(conv_block(32, 8, 3, 2))
        tower.append(nn.Flatten())
        tower.append(nn.Linear(8*((pooler_resolution//8)**2), pooler_resolution))
        tower.append(nn.ReLU())
        if cfg.MODEL.SEPSNET.CONTOUR_PREDICT:
            tower.append(nn.Linear(pooler_resolution, 2))
        else:
            tower.append(nn.Linear(pooler_resolution, 1))
        self.add_module('tower', nn.Sequential(*tower))

    def forward(self, features):
        iou_score = self.tower(features)
        return iou_score


class AuxConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        conv_block = conv_with_kaiming_uniform(None, True)  # conv relu bn
        tower = []
        tower.append(conv_block(1, 1, 3, 1))
        tower.append(nn.Conv2d(1, 1, kernel_size=1))
        self.add_module('tower', nn.Sequential(*tower))
        
    def forward(self, features):
        contour_predict = self.tower(features)
        return contour_predict