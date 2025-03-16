import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mmdet.models import LOSSES

print("### DepthLoss загружается! ###")  # Добавь эту строку

@LOSSES.register_module()
class DepthLoss(nn.Module):
    """Кастомный лосс для оценки глубины."""

    def __init__(self, depth_channels, downsample_factor, dbound, loss_weight=3.0):
        """
        Args:
            depth_channels (int): Количество дискретных каналов глубины.
            downsample_factor (int): Коэффициент даунсемплинга.
            dbound (tuple): Границы глубины (min, max, step).
            loss_weight (float): Вес лосса.
        """
        super(DepthLoss, self).__init__()
        self.depth_channels = depth_channels
        self.downsample_factor = downsample_factor
        self.dbound = dbound
        self.loss_weight = loss_weight

    def forward(self, depth_labels, depth_preds):
        """Вычисление функции потерь.

        Args:
            depth_labels (Tensor): Истинные значения глубины [B, N, H, W].
            depth_preds (Tensor): Предсказанные значения глубины [B, C, H, W].

        Returns:
            Tensor: Значение функции потерь.
        """
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none'
            ).sum() / max(1.0, fg_mask.sum())

        return self.loss_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """Даунсемплинг карты глубины.

        Args:
            gt_depths (Tensor): GT-глубина [B, N, H, W].

        Returns:
            Tensor: Даунсемпленная глубина в виде one-hot вектора.
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample_factor * self.downsample_factor)

        # Минимальное ненулевое значение в блоке
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor, W // self.downsample_factor)

        # Нормализация глубины
        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths)
        )

        # One-hot представление глубины
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.depth_channels + 1).view(
            -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()
