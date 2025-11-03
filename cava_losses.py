# -*- coding: utf-8 -*-
"""
CAVA 对齐相关损失（数值稳定 & 兼容形状；完全版）
- info_nce_align:   标准 InfoNCE，对齐 a<->v 的同索引正样本；支持 mask 权重与 tau/temperature 两种命名
- corr_diag_align:  相关性对齐（简化稳定版）
- prior_l2:         Δt 的高斯先验（如给出 mu/sigma）
- edge_hinge:       Δt 的边界铰链损失（限制在 [low, high] 的缓冲区间）
- compute_cava_losses: 汇总 CAVA 各损失（可选）
- causal_supervised_loss / causal_self_supervised_loss: 可选的因果一致性示例
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

EPS = 1e-8

# -------------------- utils --------------------
def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    """[B,T,D] or [N,D] -> [N,D]"""
    if x.ndim == 3:
        B, T, D = x.shape
        return x.reshape(B * T, D)
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"expect [B,T,D] or [N,D], got {tuple(x.shape)}")

def _mask_to_weights(mask: Optional[torch.Tensor], N: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    将 mask 统一成 [N] 的权重向量：
    - None -> 全1
    - [B,T] / [B,T,1] / [N] -> 展平为 [N]，并裁剪到 [0,1]
    """
    if mask is None:
        return torch.ones(N, device=device, dtype=dtype)
    m = mask
    if m.ndim == 3 and m.size(-1) == 1:
        m = m.squeeze(-1)
    if m.ndim == 2:
        m = m.reshape(-1)
    if m.ndim != 1:
        raise ValueError(f"mask must be [B,T], [B,T,1] or [N], got {tuple(mask.shape)}")
    m = m.to(device=device, dtype=dtype)
    m = torch.clamp(m, 0.0, 1.0)
    return m

# -------------------- losses --------------------
def info_nce_align(
    a: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tau: Optional[float] = 0.07,
    temperature: Optional[float] = None,
    normalize: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    标准 InfoNCE（同索引正样本），兼容 StrongTrainer 中的调用：
        loss_align = info_nce_align(a_aln, v_prj, mask=g, tau=...)
    也兼容 temperature 同义参数（若两者均给，以 tau 优先）。

    Args:
        a: [B,T,Da] or [N,Da]  —— 作为 “query”
        v: [B,T,Dv] or [N,Dv]  —— 作为 “key”
        mask:  [B,T] / [B,T,1] / [N]，作为样本权重（例如 CAVA 的 causal_gate）
        tau/temperature: 温度系数（越小对比分布越尖锐）
        normalize: 是否先进行 L2 归一化
        reduction: 'mean' | 'sum' | 'none'
    """
    # --- 参数别名处理 ---
    if tau is None and temperature is None:
        tau = 0.07
    if tau is None and temperature is not None:
        tau = float(temperature)
    tau = float(tau)

    # --- 展平到 [N,D] ---
    with torch.amp.autocast('cuda', enabled=False):
        a = _flatten_bt(a).float()
        v = _flatten_bt(v).float()
        if normalize:
            a = F.normalize(a, dim=-1, eps=EPS)
            v = F.normalize(v, dim=-1, eps=EPS)

        N, Da = a.shape
        Nv, Dv = v.shape
        if N != Nv:
            raise ValueError(f"InfoNCE expects same N after flatten, got {N} vs {Nv}")

        # pairwise 相似度（数值安全）
        logits = (a @ v.t()).div_(max(tau, EPS))  # [N,N]
        logits = torch.clamp(logits, min=-60.0, max=60.0)

        # 正样本标签为对角线
        target = torch.arange(N, device=logits.device)

        # 带权交叉熵：先算逐样本 CE，再按 mask 加权
        logp = F.log_softmax(logits, dim=1)                    # [N,N]
        loss_i = F.nll_loss(logp, target, reduction='none')    # [N]

        w = _mask_to_weights(mask, N, logits.device, logits.dtype)
        w_sum = torch.clamp(w.sum(), min=EPS)
        loss = (loss_i * w).sum() / w_sum

        if reduction == "sum":
            return (loss_i * w).sum()
        elif reduction == "none":
            return loss_i * w
        else:
            return loss

def corr_diag_align(
    a: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    简化的“相关性对齐”：最大化同索引相关，最小化非对角相关。
    实现为 1 - corr_diag 的权重平均，可作为 InfoNCE 的替代。
    """
    with torch.amp.autocast('cuda', enabled=False):
        a = _flatten_bt(a).float()
        v = _flatten_bt(v).float()

        a = a - a.mean(dim=0, keepdim=True)
        v = v - v.mean(dim=0, keepdim=True)
        a = F.normalize(a, dim=-1, eps=EPS)
        v = F.normalize(v, dim=-1, eps=EPS)

        C = a @ v.t()                # [N,N], 相关矩阵（已归一化）
        diag = torch.diag(C)         # [N]
        loss_i = 1.0 - diag          # 想让 diag -> 1

        if mask is not None:
            w = _mask_to_weights(mask, loss_i.numel(), loss_i.device, loss_i.dtype)
        else:
            w = torch.ones_like(loss_i)

        w_sum = torch.clamp(w.sum(), min=EPS)
        loss = (loss_i * w).sum() / w_sum

        if reduction == "sum":
            return (loss_i * w).sum()
        elif reduction == "none":
            return loss_i * w
        else:
            return loss

def prior_l2(delta: torch.Tensor, mu: Optional[float], sigma: Optional[float]) -> torch.Tensor:
    """
    Δt 的高斯先验：((delta - mu) / sigma)^2 的平均。
    若 mu/sigma 任一缺失，返回 0。
    """
    if (mu is None) or (sigma is None) or (sigma <= 0):
        return delta.new_zeros([])
    with torch.amp.autocast('cuda', enabled=False):
        z = (delta.float() - float(mu)) / float(sigma)
        return (z * z).mean()

def edge_hinge(
    delta: torch.Tensor,
    low: float,
    high: float,
    margin_ratio: float = 0.25,
) -> torch.Tensor:
    """
    边界铰链损失：鼓励 Δt 落在 [low, high] 的缓冲区内。
    在两侧各留 margin_ratio * (high-low) 的软边界。
    """
    assert high >= low, "edge_hinge: high must >= low"
    with torch.amp.autocast('cuda', enabled=False):
        L = float(low)
        H = float(high)
        m = float(margin_ratio) * (H - L)
        d = delta.float()

        left  = F.relu((L + m) - d) / (m + EPS)   # 低于软左边界的惩罚
        right = F.relu(d - (H - m)) / (m + EPS)   # 高于软右边界的惩罚
        left = torch.clamp(left, max=10.0)
        right = torch.clamp(right, max=10.0)
        return (left + right).mean()

# -------------------- optional: loss packer --------------------
def compute_cava_losses(outputs: Dict, cfg: Dict) -> Dict[str, torch.Tensor]:
    """
    计算所有CAVA相关损失，带完整的数值保护（可选）
    - 你也可以直接在 Trainer 里逐项调用（当前 StrongTrainer 即为逐项调用）
    """
    losses = {}
    with torch.amp.autocast('cuda', enabled=False):
        if cfg.get('use_infonce', True) and cfg.get('lambda_align', 0) > 0:
            v_proj = outputs.get('video_proj')
            a_aln = outputs.get('audio_aligned')
            gate = outputs.get('causal_gate')
            if v_proj is not None and a_aln is not None:
                tau = float(cfg.get('tau', 0.07))
                losses['align'] = cfg['lambda_align'] * info_nce_align(a_aln, v_proj, mask=gate, tau=tau)

        if cfg.get('lambda_prior', 0) > 0:
            delta = outputs.get('delay_frames')
            if delta is not None:
                mu = cfg.get('prior_mu', None)
                sigma = cfg.get('prior_sigma', None)
                if (mu is not None) and (sigma is not None):
                    losses['prior'] = cfg['lambda_prior'] * prior_l2(delta, mu, sigma)

        if cfg.get('lambda_edge', 0) > 0:
            delta = outputs.get('delay_frames')
            if delta is not None:
                low = cfg.get('delta_low_frames', 2.0)
                high = cfg.get('delta_high_frames', 6.0)
                margin = cfg.get('edge_margin_ratio', 0.25)
                losses['edge'] = cfg['lambda_edge'] * edge_hinge(delta, low, high, margin_ratio=margin)
    return losses

# -------------------- optional: causal consistency --------------------
def causal_supervised_loss(audio_proj: torch.Tensor,
                           video_proj: torch.Tensor,
                           class_labels: torch.Tensor,
                           cava_module,
                           weight: float = 1.0) -> torch.Tensor:
    """让分布期望 Δt 逼近类别特异 Δt_c（若存在该参数）"""
    if cava_module is None or getattr(cava_module, "class_delay", None) is None:
        return audio_proj.new_zeros([])
    scores = cava_module._corr_scores(audio_proj, video_proj)  # (B,K)
    prob = F.softmax(scores, dim=1)
    md = int(getattr(cava_module, "dist_max_delay", (prob.size(1)-1)//2))
    offsets = torch.arange(-md, md+1, device=prob.device, dtype=prob.dtype)
    exp_dt = (prob * offsets.unsqueeze(0)).sum(1)  # (B,)
    dt_c = cava_module.class_delay[class_labels.to(cava_module.class_delay.device)]
    return weight * F.mse_loss(exp_dt, dt_c)

def causal_self_supervised_loss(audio_proj: torch.Tensor,
                                video_proj: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    """无监督的 AV 一致性（InfoNCE）"""
    return info_nce_align(audio_proj, video_proj, mask=None, tau=temperature)
