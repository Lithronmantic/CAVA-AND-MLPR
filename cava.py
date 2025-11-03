# -*- coding: utf-8 -*-
"""
CAVA (Causal Audio-Visual Alignment)
- 学习延时 Δt（单位：帧），对音频序列做可微分右移，使“声先视后”
- 输出对齐后的音频序列、门控 g(t)、以及 trainer/可视化所需辅助量
- 保持与你现有 EnhancedAVTopDetector / strong_trainer 的键名兼容
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def _clamp01(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return torch.clamp(x, min=a, max=b)

def soft_shift_right(A: torch.Tensor, delta_frames: torch.Tensor) -> torch.Tensor:
    """
    对 [B,T,D] 的 A 按 Δt 右移（线性插值）。delta_frames: [B] 或标量张量。
    """
    B, T, D = A.shape
    if delta_frames.ndim == 0:
        delta_frames = delta_frames.view(1).expand(B)
    delta = delta_frames.view(B, 1, 1).clamp_min(0.0).clamp_max(max(T - 1, 0))

    n = torch.floor(delta)               # 整数部分
    alpha = (delta - n).to(A.dtype)      # 小数部分
    n = n.long()

    t = torch.arange(T, device=A.device).view(1, T, 1)
    idx0 = torch.clamp(t - n, 0, T - 1)  # t - n
    idx1 = torch.clamp(idx0 - 1, 0, T - 1)

    A0 = torch.gather(A, 1, idx0.expand(B, T, D))
    A1 = torch.gather(A, 1, idx1.expand(B, T, D))
    return (1.0 - alpha) * A0 + alpha * A1  # [B,T,D]

class LearnableDelay(nn.Module):
    """ Δt = L + (U-L) * sigmoid(theta) """
    def __init__(self, low_frames: float = 2.0, high_frames: float = 6.0, init_mid: bool = True):
        super().__init__()
        self.low = float(low_frames)
        self.high = float(high_frames)
        init = 0.0 if init_mid else -2.0
        self.theta = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, B: int) -> torch.Tensor:
        # 关闭 autocast，保证 Δt 的数值稳定
        with torch.amp.autocast('cuda', enabled=False):
            theta = torch.clamp(self.theta, -10.0, 10.0)
            delta = self.low + (self.high - self.low) * torch.sigmoid(theta)
            return delta.expand(B)  # 连续值

class CausalGate(nn.Module):
    """ 输入 concat([A_shift, V, A_shift*V]) → g∈(0,1) """
    def __init__(self, d_model: int, hidden: int = 2048, clip_min: float = 0.05, clip_max: float = 0.95):
        super().__init__()
        hidden = min(hidden, d_model * 4)
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, v: torch.Tensor, a_shift: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            v = F.normalize(v.float(), p=2, dim=-1, eps=1e-8)
            a = F.normalize(a_shift.float(), p=2, dim=-1, eps=1e-8)
            x = torch.cat([a, v, a * v], dim=-1)  # [B,T,3D]
            B, T, _ = x.shape
            g = self.net(x.view(B * T, -1)).view(B, T, 1)
            g = torch.sigmoid(torch.clamp(g, -10.0, 10.0))
            g = _clamp01(g, self.clip_min, self.clip_max)
            return g

class CAVAModule(nn.Module):
    """
    - 把 A,V 投到同一维 d_model
    - 学 Δt(帧)，对 A 做软右移
    - 产生因果门控 g（供损失&可视化；不直接混模态，交给后续融合）
    - 输出: 对齐后的 audio 序列，以及 Δt/g/相关分布等辅助量
    """
    def __init__(self, video_dim: int, audio_dim: int, d_model: int = 256,
                 delta_low_frames: float = 2.0, delta_high_frames: float = 6.0,
                 gate_clip_min: float = 0.05, gate_clip_max: float = 0.95,
                 num_classes: Optional[int] = None, dist_max_delay: int = 6):
        super().__init__()
        self.v_proj = nn.Linear(video_dim, d_model) if video_dim != d_model else nn.Identity()
        self.a_proj = nn.Linear(audio_dim, d_model) if audio_dim != d_model else nn.Identity()
        self.d_model = int(d_model)

        self.dist_max_delay = int(dist_max_delay)
        self.class_delay = nn.Parameter(torch.zeros(num_classes)) if (num_classes is not None) else None

        self.delay = LearnableDelay(delta_low_frames, delta_high_frames, init_mid=True)
        self.gate = CausalGate(d_model, hidden=2 * d_model, clip_min=gate_clip_min, clip_max=gate_clip_max)

        self.register_buffer("delta_low", torch.tensor(float(delta_low_frames)))
        self.register_buffer("delta_high", torch.tensor(float(delta_high_frames)))

        if isinstance(self.v_proj, nn.Linear):
            nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
            nn.init.zeros_(self.v_proj.bias)
        if isinstance(self.a_proj, nn.Linear):
            nn.init.xavier_uniform_(self.a_proj.weight, gain=0.5)
            nn.init.zeros_(self.a_proj.bias)

    def _corr_scores(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """计算各时延的整体相关性得分 (B, 2*md+1)"""
        if A.dim() == 2: A = A.unsqueeze(1)
        if V.dim() == 2: V = V.unsqueeze(1)
        B, Ta, Da = A.shape
        Bv, Tv, Dv = V.shape
        assert B == Bv, "Batch size mismatch"
        T = min(Ta, Tv)
        A = A[:, :T, :]
        V = V[:, :T, :]
        md = int(self.dist_max_delay)
        scores = []
        for d in range(-md, md + 1):
            if d == 0:
                s = (A * V).sum(-1).mean(1)
            elif d > 0:
                s = (A[:, :-d, :] * V[:, d:, :]).sum(-1).mean(1) if d < T else torch.zeros(B, device=A.device, dtype=A.dtype)
            else:
                dd = -d
                s = (A[:, dd:, :] * V[:, :-dd, :]).sum(-1).mean(1) if dd < T else torch.zeros(B, device=A.device, dtype=A.dtype)
            scores.append(s)
        return torch.stack(scores, dim=1)  # [B, 2*md+1]

    def get_predicted_delay(self, audio_seq: torch.Tensor, video_seq: torch.Tensor) -> torch.Tensor:
        """基于相关分布取 argmax - md 作为最可能Δt（整数帧，仅用于诊断）"""
        scores = self._corr_scores(audio_seq, video_seq)
        prob = F.softmax(scores, dim=1)
        md = int(self.dist_max_delay)
        return prob.argmax(1) - md

    def forward(self, video_seq: torch.Tensor, audio_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = video_seq.shape
        with torch.amp.autocast('cuda', enabled=False):
            v = F.layer_norm(self.v_proj(video_seq.float()), [self.d_model])
            a = F.layer_norm(self.a_proj(audio_seq.float()), [self.d_model])

            delta = self.delay(B)                 # [B] 连续
            a_shift = soft_shift_right(a, delta)  # [B,T,D]
            g = self.gate(v, a_shift)             # [B,T,1]

            # 相关分布与离散预测（基于未对齐的 a 与 v）
            scores = self._corr_scores(a, v)
            prob = F.softmax(scores, dim=1)       # [B, 2*md+1]
            md = int(self.dist_max_delay)
            pred_delay = prob.argmax(1) - md

            out = {
                "audio_for_fusion": a_shift,
                "audio_aligned": a_shift,
                "audio_proj": a,
                "video_proj": v,
                "audio_seq": a,                    # 便于上游诊断
                "causal_gate": g,
                "delay_frames": delta,             # 连续 Δt
                "delay_frames_cont": delta,        # 同义键，向后兼容
                "delta_low": float(self.delta_low.item()),
                "delta_high": float(self.delta_high.item()),
                "causal_prob": g.squeeze(-1),      # 便于直接画热力图
                "causal_prob_dist": prob,
                "pred_delay": pred_delay
            }
            return out