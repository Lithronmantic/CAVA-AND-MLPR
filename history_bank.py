# utils/history_bank.py
# -*- coding: utf-8 -*-
from typing import Iterable, Tuple, Optional, Dict
import torch

class HistoryBank:
    def __init__(self, momentum: float = 0.9):
        self.m = float(momentum)
        self.mean: Dict[int, float] = {}
        self.var:  Dict[int, float] = {}
        self.count: Dict[int, int] = {}

    @torch.no_grad()
    def update(self, ids: Iterable[int], values: torch.Tensor):
        # values: [B] on CPU/GPU
        v = values.detach().float().cpu().tolist()
        for sid, x in zip(ids, v):
            n = self.count.get(sid, 0)
            if n == 0:
                self.mean[sid] = x; self.var[sid] = 0.0; self.count[sid] = 1
            else:
                mu = self.mean[sid]
                self.mean[sid] = self.m * mu + (1 - self.m) * x
                self.var[sid]  = self.m * self.var[sid] + (1 - self.m) * (x - mu) * (x - self.mean[sid])
                self.count[sid] = n + 1

    @torch.no_grad()
    def query(self, ids: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        ms, ss = [], []
        for sid in ids:
            mu = self.mean.get(sid, 0.0)
            var= self.var.get(sid, 0.0)
            ms.append(mu)
            ss.append(max(var, 0.0) ** 0.5)
        return torch.tensor(ms).view(-1,1), torch.tensor(ss).view(-1,1)
