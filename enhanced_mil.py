# src/avtop/mil/enhanced_mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedMIL(nn.Module):
    """
    轻量可解释的 MIL 头（Top-K 选择 + 帧级分类 + 权重聚合）
    兼容要点：
      - 使用 self.training 判断训练/评估，不需要外部 is_training 参数
      - 返回键: clip_logits / seg_logits / weights / scores
      - 暴露 frame_classifier（nn.Sequential），便于 BiasInit 命中 frame_classifier[-1]
    """

    def __init__(self, d_in: int, num_classes: int, topk_ratio: float = 0.2,
                 dropout: float = 0.3, attn_temp: float = 1.0):
        super().__init__()
        assert 0.0 <= topk_ratio <= 1.0, "topk_ratio 应在 [0,1]"

        self.topk_ratio = float(topk_ratio)
        self.attn_temp = float(attn_temp)

        # 1) 异常/注意力评分网络：产生帧级 scores -> weights
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_in, 1)
        )

        # 2) 帧级分类头（暴露为 frame_classifier，方便 BiasInit 命中）
        self.frame_classifier = nn.Sequential(
            nn.Linear(d_in, max(d_in // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(d_in // 2, 1), num_classes)  # ★ BiasInit 会命中这里
        )

    @staticmethod
    def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        scores: [B,T]  ->  返回 one-hot 稀疏掩码，TopK 位置为 1，其余 0
        """
        B, T = scores.shape
        k = max(1, min(T, int(k)))
        topk_scores, topk_idx = torch.topk(scores, k, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1.0)
        # 归一化到均值为 1/k，便于加权平均时权重和=1
        mask = mask / float(k)
        return mask

    def forward(self, z: torch.Tensor):
        """
        z: [B, T, D] 融合后的时间序列特征
        returns:
            {
              'clip_logits': [B, C],
              'seg_logits':  [B, T, C],
              'weights':     [B, T],
              'scores':      [B, T]
            }
        """
        assert z.dim() == 3, f"EnhancedMIL 期望输入 [B,T,D]，得到 {tuple(z.shape)}"
        B, T, D = z.shape

        # 1) 帧级打分（scores）
        scores = self.anomaly_scorer(z).squeeze(-1)  # [B,T]

        # 2) 生成权重（weights）
        if self.training and self.topk_ratio > 0.0:
            k = int(round(T * self.topk_ratio))
            weights = self._topk_mask(scores, k)           # 稀疏 one-hot/k
        else:
            # 评估阶段用软注意力；支持温度缩放（attn_temp<1 更尖锐）
            temp = max(self.attn_temp, 1e-6)
            weights = F.softmax(scores / temp, dim=1)       # [B,T]
        # 数值安全
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # 3) 帧级分类（seg_logits）
        seg_logits = self.frame_classifier(z)               # [B,T,C]

        # 4) Clip 级：用权重对帧级 logits 进行加权聚合（而不是再过一个独立的 MLP）
        clip_logits = torch.einsum('btc,bt->bc', seg_logits, weights)

        return {
            "clip_logits": clip_logits,
            "seg_logits": seg_logits,
            "weights": weights,
            "scores": scores
        }
