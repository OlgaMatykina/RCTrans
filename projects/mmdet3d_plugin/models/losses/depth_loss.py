import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mmdet.models.losses.utils import weighted_loss

from mmdet.models import LOSSES

print("### DepthLoss загружается! ###")  # Добавь эту строку

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    fg_mask = torch.max(target, dim=1).values > 0.0
    loss = F.binary_cross_entropy(
        pred[fg_mask],
        target[fg_mask],
        reduction='none'
    ).sum() / max(1.0, fg_mask.sum())
    return loss


@LOSSES.register_module()
class DepthLoss(nn.Module):
    """Кастомный лосс для оценки глубины."""

    def __init__(self, downsample_factor, dbound, loss_weight=3.0):
        """
        Args:
            depth_channels (int): Количество дискретных каналов глубины.
            downsample_factor (int): Коэффициент даунсемплинга.
            dbound (tuple): Границы глубины (min, max, step).
            loss_weight (float): Вес лосса.
        """
        super(DepthLoss, self).__init__()
        self.downsample_factor = downsample_factor
        self.dbound = dbound
        self.loss_weight = loss_weight
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])

    def forward(self, depth_labels, depth_preds):

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_labels = depth_labels.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        

        with autocast(enabled=False):
            depth_loss = my_loss(depth_preds, depth_labels)

        return self.loss_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N, d, H//downsample, W//downsample]
        """

        if isinstance(gt_depths, list):
            gt_depths = torch.stack(gt_depths, dim=0)

        B, S, N, H, W = gt_depths.shape
        gt_depths = gt_depths.contiguous().view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor, W // self.downsample_factor, 
                                self.downsample_factor * self.downsample_factor)

        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values

        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))

        # Теперь one_hot возвращает [B*N, H, W, d] вместо (B*N*h*w, d)
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.depth_channels + 1)[..., 1:]

        # Изменяем на [B*N, d, H, W]
        gt_depths = gt_depths.permute(0, 3, 1, 2).float()

        return gt_depths
