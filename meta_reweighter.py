# models/meta_reweighter.py
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=1, keepdim=True)  # [B,1]


def _margin(p: torch.Tensor) -> torch.Tensor:
    # top1 - top2
    top2 = torch.topk(p, k=min(2, p.size(1)), dim=1).values  # [B,2]
    if top2.size(1) == 1:
        return torch.ones_like(top2[:, :1])
    return (top2[:, :1] - top2[:, 1:2])  # [B,1]


def _to2d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 1:
        return x.view(x.size(0), -1)
    if x.dim() > 2:
        dims = tuple(range(1, x.dim()))
        x = x.mean(dim=dims)
    return x  # [B,D]


def build_mlpr_features(
    teacher_prob: torch.Tensor,             # [B,C], probability
    student_feat: Optional[torch.Tensor],   # [B,D] or [B,T,D]
    history_mean: Optional[torch.Tensor] = None,  # [B,1] or [B,K] or [B,T,1]
    history_std: Optional[torch.Tensor] = None,   # same
    cava_gate_mean: Optional[torch.Tensor] = None,# same
    use_prob_vector: bool = False
) -> torch.Tensor:
    """
    组合出 Meta-Weight-Net 的输入特征（每样本一行）
    基本项：max prob / entropy / margin
    可选项：student feat 范数、历史均值/方差、CAVA门控均值、完整prob向量
    """
    B, C = teacher_prob.shape
    maxp = teacher_prob.max(dim=1, keepdim=True).values
    ent = _entropy(teacher_prob)
    mar = _margin(teacher_prob)

    student_feat = _to2d(student_feat)
    history_mean = _to2d(history_mean)
    history_std = _to2d(history_std)
    cava_gate_mean = _to2d(cava_gate_mean)

    feats = [maxp, ent, mar]
    if student_feat is not None:
        feats.append(student_feat.norm(p=2, dim=1, keepdim=True))  # [B,1]
    if history_mean is not None:
        feats.append(history_mean)
    if history_std is not None:
        feats.append(history_std)
    if cava_gate_mean is not None:
        feats.append(cava_gate_mean)
    if use_prob_vector:
        feats.append(teacher_prob)

    x = torch.cat(feats, dim=1)  # [B,D_in]
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    return x


class MetaReweighter(nn.Module):
    """
    简单稳定的权重网络：LN -> FC -> ReLU -> Drop -> FC -> ReLU -> Drop -> FC -> Sigmoid -> [w_min,w_max]
    """
    def __init__(self, input_dim: int, hidden=(128, 64), w_clip=(0.05, 0.95), dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, h1), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )
        self.w_min = float(w_clip[0])
        self.w_max = float(w_clip[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.net(x))  # [B,1]
        return w.clamp(min=self.w_min, max=self.w_max)
