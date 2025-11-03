# losses/ssl_losses.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F

def kl_divergence(student_logits: torch.Tensor,
                  teacher_prob: torch.Tensor,
                  T: float = 1.0,
                  reduction: str = "mean") -> torch.Tensor:
    """
    KL( p_s || p_t )，其中 p_s = softmax(student_logits/T)
    teacher_prob 假定已是概率分布（会做归一 & clamp）
    """
    eps = 1e-8
    t = teacher_prob.clamp_min(eps)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(eps)
    log_ps = F.log_softmax(student_logits / T, dim=1)
    ps = log_ps.exp()
    # KL = sum ps * (log ps - log pt)
    kl = (ps * (log_ps - (t + eps).log())).sum(dim=1)
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl  # [B]

def soft_ce(student_logits: torch.Tensor,
            teacher_prob: torch.Tensor,
            T: float = 1.0,
            reduction: str = "mean") -> torch.Tensor:
    # 交叉熵 CE(teacher || student)
    log_ps = F.log_softmax(student_logits / T, dim=1)
    t = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ce = -(t * log_ps).sum(dim=1)
    return ce.mean() if reduction == "mean" else ce

def hard_ce(student_logits: torch.Tensor,
            hard_targets: torch.Tensor,
            reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(student_logits, hard_targets, reduction=reduction)

def ramp_up(curr_epoch: int, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return 1.0
    t = max(0.0, min(1.0, curr_epoch / float(ramp_epochs)))
    # 指数/高斯 ramp 都可，取平滑三次多项式
    return 3*t**2 - 2*t**3
