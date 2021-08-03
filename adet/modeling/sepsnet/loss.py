import torch
from torch.nn import functional as F


def dice_coefficient(x, target):
    smooth = 1
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1)
    coefficient = (2 * intersection + smooth) / (union + smooth)
    return coefficient


def contour_calc(predicts, targets):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1,
         -1,  8, -1,
         -1, -1, -1],
        dtype=torch.float32, device=predicts.device).reshape(1, 1, 3, 3).requires_grad_(False)

    predicts_mask = (torch.sigmoid(predicts) > 0.5).to(device=predicts.device)
    predicts_contour = F.conv2d(predicts_mask.unsqueeze(1).float(), laplacian_kernel, padding=1)
    predicts_contour = predicts_contour.clamp(min=0)
    predicts_contour[predicts_contour > 0] = 1
    predicts_contour[predicts_contour <= 0] = 0

    targets_contour = F.conv2d(targets.unsqueeze(1).float(), laplacian_kernel, padding=1).detach()
    targets_contour = targets_contour.clamp(min=0)
    targets_contour[targets_contour > 0] = 1
    targets_contour[targets_contour <= 0] = 0

    if predicts_contour.shape[-1] != targets_contour.shape[-1]:
        targets_contour = F.interpolate(
            targets_contour, predicts_contour.shape[2:], mode='nearest')
    coefficient = dice_coefficient(predicts_contour, targets_contour)
    return coefficient


def contour_loss_func(predicts, targets, resize_ratio=1, predict_conv=True):
    """
    Args:
        predicts (Tensor): A tensor of shape (B, H, W) 
        targets (Tensor): A tensor of shape (B, H, W) 
    """
    laplacian_kernel = torch.tensor(
        [-1, -1, -1,
         -1,  8, -1,
         -1, -1, -1],
        dtype=torch.float32, device=predicts.device).reshape(1, 1, 3, 3).requires_grad_(False)

    predicts = F.interpolate(predicts.unsqueeze(1), scale_factor=resize_ratio)
    if predict_conv:
        predicts_contour = F.conv2d(torch.sigmoid(predicts), laplacian_kernel, padding=1)
    else:
        predicts_contour = torch.sigmoid(predicts)

    targets = F.interpolate(targets.unsqueeze(1).float(), predicts_contour.shape[2:])
    targets_contour = F.conv2d(targets, laplacian_kernel, padding=1).detach()
    targets_contour[targets_contour > 0] = 1
    targets_contour[targets_contour <= 0] = 0

    coefficient = dice_coefficient(predicts_contour, targets_contour)
    dice_loss = 1-coefficient
    return dice_loss


def contour_raw_loss_func(predicts, targets_contour, resize_ratio=1):
    """
    Args:
        predicts (Tensor): A tensor of shape (B, H, W) 
        targets_contour: gt for contour of shape (B, H, W)
    """
    predicts = F.interpolate(predicts.unsqueeze(1), scale_factor=resize_ratio)
    predicts_contour = predicts

    targets_contour = targets_contour.unsqueeze(1).float()
    targets_contour = F.interpolate(targets_contour, size=predicts_contour.shape[-2:])
    coefficient = dice_coefficient(predicts_contour.sigmoid(), targets_contour)
    dice_loss = 1-coefficient

    return dice_loss


def iou_calc(predicts, targets):
    """
    Args:
        predicts (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        targets (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    eps = 1e-5
    predicts_mask = (torch.sigmoid(predicts) > 0.5).to(device=predicts.device)

    intersection = torch.logical_and(predicts_mask, targets).sum(dim=(1,2))
    union = torch.logical_or(predicts_mask, targets).sum(dim=(1,2)) + eps
    return intersection/union


def iou_loss_func(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = ((x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) - intersection) + eps
    jaccard = intersection/union
    return 1-jaccard