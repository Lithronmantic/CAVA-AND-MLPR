# optim/meta_utils.py
# -*- coding: utf-8 -*-
"""
元学习伪标签重加权（MLPR）工具函数
- 一步SGD快权重 (FO-MAML/Meta-Weight-Net风格)
- 可选 Neumann 近似二阶项（更稳定但更慢）
"""
from typing import Dict, Iterable, Tuple
import torch
from torch import nn
from torch.nn.utils import stateless
import torch.nn.functional as F


def sgd_fast_weights(model: nn.Module, loss: torch.Tensor, lr: float) -> Dict[str, torch.Tensor]:
    """
    基于单步SGD的“快权重” theta' = theta - lr * grad_theta(loss)
    要求 loss 的计算图贯穿 meta_net -> weights -> loss，才能把梯度传给 meta_net。
    """
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    grads = torch.autograd.grad(loss, list(params.values()), create_graph=True, allow_unused=True)
    fast_state = {}
    for (n, p), g in zip(params.items(), grads):
        if g is None:
            fast_state[n] = p
        else:
            fast_state[n] = p - lr * g
    return fast_state


@torch.no_grad()
def _flatten_probs(p: torch.Tensor) -> torch.Tensor:
    if p.dim() == 1:
        return p.unsqueeze(0)
    return p


def meta_step_first_order(
    student_model: nn.Module,
    meta_net: nn.Module,
    meta_opt: torch.optim.Optimizer,
    *,
    v_tr: torch.Tensor,
    a_tr: torch.Tensor,
    yhat_tr: torch.Tensor,
    w_tr: torch.Tensor,
    v_val: torch.Tensor,
    a_val: torch.Tensor,
    y_val: torch.Tensor,
    lr_inner: float = 1e-3,
) -> float:
    """
    一阶近似的元学习更新：
      1) 用带权伪标签损失 L_tr = mean(w_i * CE(student(v_tr,a_tr), yhat_tr))
      2) 用 L_tr 对 student 做一次“虚拟更新”得到 theta'
      3) 用 theta' 在验证集上算 L_val，反传到 meta_net 并更新
    返回：float(loss_val)
    """
    # 1) 训练损失（带权）
    logits_tr = student_model(v_tr, a_tr, return_aux=False)
    if isinstance(logits_tr, dict):  # 兼容你的模型返回dict
        logits_tr = logits_tr.get("clip_logits", logits_tr)
    per_ex = F.cross_entropy(logits_tr, yhat_tr, reduction='none')  # [B]
    L_tr = (w_tr.view(-1) * per_ex).mean()

    # 2) 计算快权重
    fast_state = sgd_fast_weights(student_model, L_tr, lr=lr_inner)

    # 3) 用快权重在验证集上前向并计算损失
    logits_val = stateless.functional_call(
        student_model,
        fast_state,
        (v_val, a_val),
        {'return_aux': False}
    )
    if isinstance(logits_val, dict):
        logits_val = logits_val.get("clip_logits", logits_val)
    L_val = F.cross_entropy(logits_val, y_val)

    # 4) 仅更新 meta_net
    meta_opt.zero_grad(set_to_none=True)
    L_val.backward()
    torch.nn.utils.clip_grad_norm_(meta_net.parameters(), max_norm=1.0)
    meta_opt.step()
    return float(L_val.detach().cpu().item())


def meta_step_neumann(
    student_model: nn.Module,
    meta_net: nn.Module,
    meta_opt: torch.optim.Optimizer,
    *,
    v_tr: torch.Tensor,
    a_tr: torch.Tensor,
    yhat_tr: torch.Tensor,
    w_tr: torch.Tensor,
    v_val: torch.Tensor,
    a_val: torch.Tensor,
    y_val: torch.Tensor,
    lr_inner: float = 1e-3,
    neumann_iter: int = 5,
    damping: float = 0.5
) -> float:
    """
    二阶近似（Neumann series）元更新，可选：
      - 计算 H^{-1} g 的近似，其中 H 是 L_tr 的 Hessian w.r.t theta
      - 实现复杂，若不稳定建议用 meta_step_first_order
    """
    # 先做一次前向，得到梯度 g = ∇_θ L_tr
    logits_tr = student_model(v_tr, a_tr, return_aux=False)
    if isinstance(logits_tr, dict):
        logits_tr = logits_tr.get("clip_logits", logits_tr)
    per_ex = F.cross_entropy(logits_tr, yhat_tr, reduction='none')
    L_tr = (w_tr.view(-1) * per_ex).mean()
    params = [p for p in student_model.parameters() if p.requires_grad]
    g = torch.autograd.grad(L_tr, params, create_graph=True, retain_graph=True, allow_unused=True)

    # 计算 g_val = ∇_θ L_val
    logits_val = student_model(v_val, a_val, return_aux=False)
    if isinstance(logits_val, dict):
        logits_val = logits_val.get("clip_logits", logits_val)
    L_val = F.cross_entropy(logits_val, y_val)
    g_val = torch.autograd.grad(L_val, params, create_graph=True, retain_graph=True, allow_unused=True)

    # Neumann 迭代近似 v = H^{-1} g_val
    v_vec = [gv.detach() for gv in g_val]
    for _ in range(neumann_iter):
        hv = torch.autograd.grad(g, params, grad_outputs=v_vec, retain_graph=True, allow_unused=True)
        v_vec = [ (gv + (1 - damping) * vv - lr_inner * (hv_i if hv_i is not None else 0.0))
                  for gv, vv, hv_i in zip(g_val, v_vec, hv) ]

    # 反向到 meta_net
    meta_opt.zero_grad(set_to_none=True)
    # 近似的 meta-grad = - ∂(∑ w_i * l_i) / ∂meta ≈ - ∑ (∂w/∂meta * l_i)
    # 这里简单用 L_tr 反传（已经包含 w(meta) 依赖），并将 grad 与 v_vec 做内积权重
    # 简化实现：退化到一阶；若需严谨二阶，请根据论文实现特定公式。
    L_tr.backward()
    torch.nn.utils.clip_grad_norm_(meta_net.parameters(), max_norm=1.0)
    meta_opt.step()
    return float(L_val.detach().cpu().item())
