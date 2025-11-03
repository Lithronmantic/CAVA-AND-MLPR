# utils/teacher_ema.py
# -*- coding: utf-8 -*-
from copy import deepcopy
import torch
import torch.nn.functional as F

class EMATeacher:
    def __init__(self, student_model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.model = deepcopy(student_model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_model: torch.nn.Module):
        d = self.decay
        for p_t, p_s in zip(self.model.parameters(), student_model.parameters()):
            p_t.data.mul_(d).add_(p_s.data, alpha=(1.0 - d))

        # buffers（BN 等）对齐
        for b_t, b_s in zip(self.model.buffers(), student_model.buffers()):
            b_t.copy_(b_s)

    @torch.no_grad()
    def predict_prob(self, v, a) -> torch.Tensor:
        out = self.model(v, a, return_aux=False)
        logits = out["clip_logits"] if isinstance(out, dict) else out
        return F.softmax(logits, dim=1)
