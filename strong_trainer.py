# -*- coding: utf-8 -*-
"""
StrongTrainer (修复+增强版)
- Student & EMA Teacher 双路评测（取更优者计做 best）
- EMA 衰减分段（ema_decay_init→ema_decay，在 warmup 内线性过渡）
- 伪标签阈值更保守：全局阈值不超过 teacher p90_ema 的 0.9×（下界 thr_min）
- 分布对齐 / 类阈值：推迟到 p90_ema>0.35 或 epoch>warmup+2 再启用
- CAVA: 使用连续 Δt 参与 prior/edge；对齐损失稳定性增强
"""
import os, json, math, random, time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from cava_losses import info_nce_align, corr_diag_align, prior_l2, edge_hinge
from meta_reweighter import MetaReweighter, build_mlpr_features
from ssl_losses import ramp_up
from history_bank import HistoryBank
from teacher_ema import EMATeacher
from meta_utils import meta_step_first_order
from dataset import AVFromCSV, safe_collate_fn
try:
    from dataset import safe_collate_fn_with_ids
except Exception:
    def safe_collate_fn_with_ids(batch):
        return safe_collate_fn(batch)

from enhanced_detector import EnhancedAVTopDetector

# AMP 兼容封装
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    AMP_DEVICE_ARG = True
    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(device_type, enabled=enabled, dtype=dtype)
    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(device_type, enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    AMP_DEVICE_ARG = False
    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(enabled=enabled)
    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(enabled=enabled)

# Focal CE（保留你的实现）
class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        with amp_autocast('cuda', enabled=False):
            logits_f32 = torch.clamp(logits.float(), min=-30, max=30)
            ce = F.cross_entropy(
                logits_f32, targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none"
            )
            with torch.no_grad():
                pt = F.softmax(logits_f32, dim=1).gather(1, targets.view(-1, 1)).squeeze(1)
                pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
                focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * ce
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return ce.mean()
            return loss.mean()

def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class StrongTrainer:
    def __init__(self, cfg: Dict[str, Any], out_dir: str):
        self.cfg = cfg
        self.out_dir = Path(out_dir); (self.out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        _set_seed(int(cfg.get("seed", 42)))

        # AMP
        self.amp_enabled = bool(cfg.get('training', {}).get('amp', True) and self.device.type == 'cuda')
        self.scaler = AmpGradScaler(self.device_type, enabled=self.amp_enabled)
        try:
            if self.amp_enabled and hasattr(self.scaler, '_init_scale'):
                init_scale = cfg.get("training", {}).get("amp_init_scale", 512.0)
                self.scaler._scale = torch.tensor(float(init_scale), dtype=torch.float32, device=self.device)
        except Exception:
            pass
        self.amp_disable_epoch = int(cfg.get("training", {}).get("amp_disable_epoch", 15))
        self.original_amp_enabled = self.amp_enabled
        self.grad_explosion_count = 0; self.max_grad_explosion = 3

        # Data
        data_cfg = cfg["data"]
        self.C = int(data_cfg["num_classes"]); self.num_classes = self.C
        self.class_names = list(data_cfg["class_names"]); root = data_cfg.get("data_root", "")
        vcfg = cfg.get("video", {}); acfg = cfg.get("audio", {})
        l_csv = data_cfg["labeled_csv"]; v_csv = data_cfg["val_csv"]; u_csv = data_cfg.get("unlabeled_csv")
        self.ds_l = AVFromCSV(l_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=False)
        self.ds_v = AVFromCSV(v_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=False)
        self.ds_u = AVFromCSV(u_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=True) if (
            cfg.get("training", {}).get("use_ssl", False) and u_csv) else None

        # 标注先验
        self.stats = self._scan_stats(self.ds_l)
        (self.out_dir / 'stats').mkdir(exist_ok=True, parents=True)
        (self.out_dir / 'stats' / 'class_stats.json').write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2), encoding='utf-8')

        sampler = None
        if data_cfg.get("sampler", "").lower() == "weighted":
            inv_freq = np.array(self.stats["inv_freq"], dtype=np.float32)
            sampler = self._build_sampler(self.ds_l, inv_freq)
            print("[DataLoader] 使用 WeightedRandomSampler")

        tr = cfg.get("training", {})
        self.bs = int(tr.get("batch_size", 16))
        pin_mem = (self.device.type == 'cuda')
        def _to(nw, default=60): return 0 if int(nw) == 0 else default
        self.loader_l = DataLoader(
            self.ds_l, batch_size=self.bs, sampler=sampler, shuffle=(sampler is None),
            num_workers=int(data_cfg.get("num_workers_train", 0)), pin_memory=pin_mem,
            drop_last=True, collate_fn=safe_collate_fn, timeout=_to(data_cfg.get("num_workers_train", 0)),
            persistent_workers=(int(data_cfg.get("num_workers_train", 0)) > 0)
        )
        self.loader_v = DataLoader(
            self.ds_v, batch_size=self.bs, shuffle=False,
            num_workers=int(data_cfg.get("num_workers_val", 0)), pin_memory=pin_mem,
            drop_last=False, collate_fn=safe_collate_fn, timeout=_to(data_cfg.get("num_workers_val", 0)),
            persistent_workers=(int(data_cfg.get("num_workers_val", 0)) > 0)
        )
        self.loader_u = None
        if self.ds_u is not None:
            self.loader_u = DataLoader(
                self.ds_u, batch_size=self.bs, shuffle=True,
                num_workers=int(data_cfg.get("num_workers_unl", 0)), pin_memory=pin_mem,
                drop_last=True, collate_fn=safe_collate_fn_with_ids, timeout=_to(data_cfg.get("num_workers_unl", 0)),
                persistent_workers=(int(data_cfg.get("num_workers_unl", 0)) > 0)
            )

        # Model
        model_cfg = dict(cfg.get("model", {})); model_cfg["num_classes"] = self.C
        fusion_cfg = model_cfg.get("fusion", cfg.get("fusion", {}))
        self.model = EnhancedAVTopDetector({"model": model_cfg, "fusion": fusion_cfg, "cava": cfg.get("cava", {})}).to(self.device)

        # Bias init
        if bool(cfg.get("model", {}).get("init_bias", False)):
            self._init_bias(self.model, self.stats["pi"])
        else:
            print("[BiasInit] disabled by config")

        # Loss
        loss_cfg = cfg.get("loss", {})
        self.loss_name = loss_cfg.get("name", "ce").lower()
        cw = loss_cfg.get("class_weights", None)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=self.device) if cw is not None else None
        if self.loss_name == "focal_ce":
            print("Using Focal CrossEntropy (single-label)")
            self.criterion = FocalCrossEntropy(
                gamma=loss_cfg.get("gamma", 2.0),
                label_smoothing=loss_cfg.get("label_smoothing", 0.05),
                class_weights=class_weights
            ).to(self.device)
        else:
            print("Using CrossEntropy (single-label)")
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=loss_cfg.get("label_smoothing", 0.05)
            ).to(self.device)

        # Optim
        self.epochs = int(tr.get("num_epochs", 20))
        base_lr = float(tr.get("learning_rate", 5e-5))
        bb_mult = float(tr.get("backbone_lr_mult", 0.1))
        self.wd = float(tr.get("weight_decay", 1e-3))
        self.grad_clip = float(tr.get("grad_clip_norm", 0.5))

        head_params, bb_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if "video_backbone" in n or "audio_backbone" in n: bb_params.append(p)
            else: head_params.append(p)
        print(f"[OPT] Param groups: head={len(head_params)} @ {base_lr:.6f}, backbone={len(bb_params)} @ {base_lr * bb_mult:.6f}")
        self.opt = optim.AdamW(
            [{"params": head_params, "lr": base_lr}, {"params": bb_params, "lr": base_lr * bb_mult}],
            weight_decay=self.wd
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs, eta_min=1e-7)

        print(f"[AMP] enabled={self.amp_enabled}, device_type={self.device_type}, init_scale={getattr(self.scaler, '_init_scale', 'N/A')}")

        # stats
        self.nan_count = 0; self.total_steps = 0; self.meta_fail_count = 0

        # MLPR
        self.mlpr_cfg = dict(cfg.get("mlpr", {})); self.mlpr_enabled = bool(self.mlpr_cfg.get("enabled", False))
        self.teacher_ema = EMATeacher(self.model, decay=float(self.mlpr_cfg.get("ema_decay", 0.999))) if self.mlpr_enabled else None

        use_hist = bool(self.mlpr_cfg.get("use_history_stats", True))
        use_cava = bool(self.mlpr_cfg.get("use_cava_signal", True))
        use_prob_vec = bool(self.mlpr_cfg.get("use_prob_vector", False))
        feat_dim = 3 + 1 + (2 if use_hist else 0) + (1 if use_cava else 0) + (self.C if use_prob_vec else 0)
        self.meta = MetaReweighter(
            input_dim=feat_dim, hidden=(128, 64),
            w_clip=tuple(self.mlpr_cfg.get("weight_clip", [0.05, 0.95])), dropout=0.1
        ).to(self.device) if self.mlpr_enabled else None
        self.meta_opt = optim.Adam(self.meta.parameters(), lr=float(self.mlpr_cfg.get("meta_lr", 1e-4))) if self.mlpr_enabled else None
        self.hist_bank = HistoryBank(momentum=float(self.mlpr_cfg.get("history_momentum", 0.9))) if (self.mlpr_enabled and use_hist) else None
        self._mlpr_flags = {"use_hist": use_hist, "use_cava": use_cava, "use_prob_vec": use_prob_vec}
        self._mlpr_tempT = float(self.mlpr_cfg.get("T", 1.0))
        self._mlpr_lambda_u = float(self.mlpr_cfg.get("lambda_u", 0.5))
        self._mlpr_ramp_epochs = int(self.mlpr_cfg.get("ramp_up_epochs", 5))
        self._mlpr_meta_interval = int(self.mlpr_cfg.get("meta_interval", 20))
        self._mlpr_inner_lr = float(self.mlpr_cfg.get("inner_lr", base_lr))

        # SSL
        tr_ssl = cfg.get("training", {})
        self.use_ssl = bool(tr_ssl.get("use_ssl", False) and self.ds_u is not None)
        self.teacher = None
        self.ema_decay = 0.999; self.ssl_warmup = 5; self.ssl_final_thresh = 0.9; self.ssl_temp = 1.0; self.lambda_u = 1.0

        ssl_cfg = dict(cfg.get("ssl", {}))
        self._use_dist_align = bool(ssl_cfg.get("use_dist_align", True))
        self._use_cls_threshold = bool(ssl_cfg.get("use_cls_threshold", True))
        self._thr_min = float(ssl_cfg.get("thr_min", 0.05))
        self._cls_thr_momentum = float(ssl_cfg.get("cls_thr_momentum", 0.9))
        self._cls_conf_ema = torch.full((self.C,), 0.5, device=self.device)
        self._cls_thr = torch.full((self.C,), self.ssl_final_thresh, device=self.device)

        # EMA 衰减分段参数
        self.ema_decay_init = float(ssl_cfg.get("ema_decay_init", 0.99))
        if self.use_ssl:
            self.ema_decay = float(ssl_cfg.get("ema_decay", 0.999))
            self.ssl_warmup = int(ssl_cfg.get("warmup_epochs", 5))
            self.ssl_final_thresh = float(ssl_cfg.get("final_thresh", 0.9))
            self.ssl_temp = float(ssl_cfg.get("consistency_temp", 1.0))
            self.lambda_u = float(ssl_cfg.get("lambda_u", 1.0))

            self.teacher = EnhancedAVTopDetector({"model": model_cfg, "fusion": fusion_cfg, "cava": cfg.get("cava", {})}).to(self.device)
            self.teacher.load_state_dict(self.model.state_dict(), strict=True)
            for p in self.teacher.parameters(): p.requires_grad = False
            self.teacher.eval()
            print(f"[SSL] EMA={self.ema_decay} warmup={self.ssl_warmup} T={self.ssl_temp} thr*={self.ssl_final_thresh} λu={self.lambda_u}")

        # 评测策略：auto/student/teacher
        self.eval_with_ema_mode = str(ssl_cfg.get("eval_with_ema", "auto")).lower()

        self.best_f1 = -1.0
        self.cava_cfg = dict(cfg.get("cava", {}))
        self.cava_enabled = bool(self.cava_cfg.get("enabled", False))

        self._print_startup_banner()
        trcfg = cfg.get("training", {})
        self.early_stop_patience = int(trcfg.get("early_stop_patience", 0))
        self.no_improve = 0
        self._pi = torch.tensor(self.stats["pi"], dtype=torch.float32, device=self.device)

        # Teacher 置信度跟踪
        self._teach_p90_ema = 0.2

    # 打印横幅
    def _print_startup_banner(self):
        def _name(x):
            try: return type(x).__name__
            except: return str(type(x))
        vbb = getattr(self.model, 'video_backbone', None)
        abb = getattr(self.model, 'audio_backbone', None)
        print("┌─ Model/Fusion")
        print(f"│  fusion_type         = {getattr(self.model, 'fusion_type', 'n/a')}")
        print(f"│  video_dim, audio_dim= {self.model.video_dim}, {self.model.audio_dim}")
        print(f"│  fusion_dim          = {self.model.fusion_dim}")
        print(f"│  num_classes         = {self.num_classes}")
        print(f"│  backbones           = Video<{_name(vbb)}>, Audio<{_name(abb)}>")
        print("├─ CAVA (Causal Align)")
        print(f"│  enabled             = {self.cava_enabled}")
        if self.cava_enabled:
            print(f"│  d_model             = {self.cava_cfg.get('d_model', self.model.fusion_dim)}")
            print(f"│  delta_range(frames) = [{self.cava_cfg.get('delta_low_frames', 2)}, {self.cava_cfg.get('delta_high_frames', 6)}]")
            print(f"│  use_infonce         = {self.cava_cfg.get('use_infonce', True)}")
            print(f"│  tau                 = {self.cava_cfg.get('tau', 0.07)}")
            print(f"│  λ_align/prior/edge  = {self.cava_cfg.get('lambda_align', 0.05)}/"
                  f"{self.cava_cfg.get('lambda_prior', 0.0)}/{self.cava_cfg.get('lambda_edge', 0.01)}")
        print("├─ SSL (Teacher-Student EMA)")
        print(f"│  enabled             = {self.use_ssl}")
        if self.use_ssl:
            print(f"│  ema_decay_init/final= {self.ema_decay_init}/{self.ema_decay}")
            print(f"│  warmup_epochs       = {self.ssl_warmup}")
            print(f"│  final_thresh        = {self.ssl_final_thresh}")
            print(f"│  consistency_temp    = {self.ssl_temp}")
            print(f"│  lambda_u            = {self.lambda_u}")
            print(f"│  dist_align / cls_thr= {self._use_dist_align} / {self._use_cls_threshold}")
            print(f"│  eval mode           = {self.eval_with_ema_mode}")
        print("├─ MLPR (Meta Reweight)")
        print(f"│  enabled             = {self.mlpr_enabled}")
        if self.mlpr_enabled:
            flags = self._mlpr_flags
            inputs = [
                ("teacher_top1_conf", True),
                ("teacher_entropy",   True),
                ("teacher_margin",    True),
                ("student_fusion_feat", True),
                ("history_mean",      flags.get("use_hist", True)),
                ("history_std",       flags.get("use_hist", True)),
                ("cava_gate_mean",    flags.get("use_cava", True)),
            ]
            n_base = sum(int(v) for _, v in inputs)
            prob_vec = int(flags.get("use_prob_vec", False))
            print(f"│  inputs(7 base)      = {n_base}  [{', '.join(n for n, v in inputs if v)}]")
            print(f"│  +prob_vector(C?)    = {bool(prob_vec)}")
            print(f"│  meta_input_dim      = {getattr(self.meta, 'input_dim', 'N/A') if self.meta is not None else 'N/A'}")
            print(f"│  inner_lr/meta_lr    = {self._mlpr_inner_lr}/{self.mlpr_cfg.get('meta_lr', 1e-4)}")
            print(f"│  T/lambda_u          = {self._mlpr_tempT}/{self._mlpr_lambda_u}")
            print(f"│  interval/ramp_epochs= {self._mlpr_meta_interval}/{self._mlpr_ramp_epochs}")
        print("├─ Data/Loss")
        print(f"│  sampler             = {'WeightedRandomSampler' if self.loader_l.sampler is not None else 'shuffle'}")
        print(f"│  batch_size          = {self.bs}")
        print(f"│  loss                = {'FocalCE' if self.loss_name=='focal_ce' else 'CrossEntropy'}")
        print("└────────────────────────────────────────────────────────────────")

    # 统计
    def _scan_stats(self, ds_l) -> Dict[str, Any]:
        C = self.C
        counts = np.zeros(C, dtype=np.int64)
        n = len(ds_l)
        for i in range(n):
            try:
                item = ds_l[i]
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    y = item[2]
                elif isinstance(item, dict) and 'label' in item:
                    y = item['label']
                else:
                    continue
                if torch.is_tensor(y):
                    y = y.detach().cpu()
                    if y.ndim == 0:
                        idx = int(y.item())
                    elif y.ndim == 1:
                        idx = int(y.argmax().item())
                    elif y.ndim == 2:
                        idx = int(y.argmax(dim=1)[0].item())
                    else:
                        continue
                else:
                    idx = int(y)
                if 0 <= idx < C:
                    counts[idx] += 1
            except Exception:
                continue
        total = counts.sum()
        pi = (counts / total) if total > 0 else np.ones(C, dtype=np.float32) / C
        inv = 1.0 / np.clip(counts.astype(np.float32), 1.0, None); inv = inv / inv.mean()
        return {
            "counts": counts.tolist(),
            "pi": pi.astype(np.float32).tolist(),
            "inv_freq": inv.astype(np.float32).tolist(),
            "total": int(total),
        }

    def _forward(self, v: torch.Tensor, a: torch.Tensor):
        return self.model(v, a, return_aux=True)

    def _reset_scaler_if_needed(self):
        if hasattr(self, 'scaler') and self.scaler is not None and self.scaler.is_enabled():
            old_scale = self.scaler.get_scale() if hasattr(self.scaler, 'get_scale') else 1024.0
            self.scaler = AmpGradScaler(self.device_type, enabled=True)
            new_scale = max(float(old_scale) * 0.5, 2.0)
            self.scaler._scale = torch.tensor(new_scale, dtype=torch.float32, device=self.device)
            self.scaler._growth_tracker = torch.tensor(0, dtype=torch.int32, device=self.device)
            print(f"✓ GradScaler已重置: scale {old_scale:.0f} -> {new_scale:.0f}")
        if hasattr(self, 'opt') and self.opt is not None:
            self.opt.zero_grad(set_to_none=True)

    def _thr_at(self, epoch: int) -> float:
        if not self.use_ssl: return 0.0
        if epoch <= self.ssl_warmup:
            return self.ssl_final_thresh * (epoch / max(1, self.ssl_warmup))
        else:
            return self.ssl_final_thresh

    def _ema_update(self, frac_in_epoch: float = 1.0):
        if self.teacher is None: return
        # ema 衰减分段：warmup 内从 ema_decay_init 过渡到 ema_decay
        k = min(1.0, max(0.0, (self.current_epoch - 1 + frac_in_epoch) / max(1, self.ssl_warmup)))
        ema_now = self.ema_decay_init * (1 - k) + self.ema_decay * k
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
                t_param.data.mul_(ema_now).add_(s_param.data, alpha=1.0 - ema_now)

    def _init_bias(self, model, pi):
        pi_tensor = torch.tensor(pi, dtype=torch.float32, device=self.device)
        def _try_set_bias(linear: nn.Linear, tag: str):
            if isinstance(linear, nn.Linear) and linear.bias is not None:
                with torch.no_grad():
                    log_pi = torch.log(torch.clamp(pi_tensor, min=1e-8)).to(linear.bias.device)
                    linear.bias.copy_(log_pi)
                print(f"[BiasInit] 已初始化 {tag} bias，范围[{log_pi.min():.3f}, {log_pi.max():.3f}]")
                return True
            return False

        hit = False
        try:
            classifier = None
            if hasattr(model, 'classifier'):
                classifier = model.classifier
            elif hasattr(model, 'head'):
                classifier = model.head
            elif hasattr(model, 'fc'):
                classifier = model.fc
            if classifier is not None:
                last_linear = None
                if isinstance(classifier, nn.Sequential):
                    for layer in reversed(list(classifier)):
                        if isinstance(layer, nn.Linear):
                            last_linear = layer; break
                elif isinstance(classifier, nn.Linear):
                    last_linear = classifier
                if last_linear is not None:
                    if _try_set_bias(last_linear, "主分类层(classifier/head/fc)"):
                        hit = True
        except Exception as e:
            print(f"[BiasInit] 主分类层初始化失败: {e}")

        try:
            if hasattr(model, "mil_head") and hasattr(model.mil_head, "frame_classifier"):
                seq = model.mil_head.frame_classifier
                last = None
                if isinstance(seq, nn.Sequential):
                    for m in reversed(list(seq)):
                        if isinstance(m, nn.Linear):
                            last = m; break
                elif isinstance(seq, nn.Linear):
                    last = seq
                if last is not None:
                    if _try_set_bias(last, "MIL 头最终层(frame_classifier)"):
                        hit = True
        except Exception as e:
            print(f"[BiasInit] MIL 头 bias 初始化失败: {e}")

        if not hit:
            print("[BiasInit] 未命中任何可初始化的线性层（classifier/head/fc 或 MIL 头）")

    # meta update（保留现有一阶近似）
    def _meta_update_step(self, step_count: int):
        if not self.mlpr_enabled or self.meta is None or self.meta_opt is None: return
        try:
            val_iter = iter(self.loader_v); val_batch = next(val_iter)
            if len(val_batch) == 4: v_val, a_val, y_val, _ = val_batch
            else: v_val, a_val, y_val = val_batch
            v_val = v_val.to(self.device); a_val = a_val.to(self.device)
            y_val = y_val.argmax(dim=1).to(self.device) if y_val.ndim == 2 else y_val.to(self.device)

            with amp_autocast(self.device_type, enabled=False):
                if hasattr(self, '_last_train_batch'):
                    v_train, a_train, y_train = self._last_train_batch
                    v_train = v_train.float().detach(); a_train = a_train.float().detach()
                    v_val = v_val.float(); a_val = a_val.float()

                    self.model.eval()
                    out_train = self._forward(v_train, a_train)
                    if out_train is None or "clip_logits" not in out_train:
                        print("[Meta] 训练前向失败"); return
                    logits_train = out_train["clip_logits"]

                    train_loss = self.criterion(logits_train, y_train)
                    if hasattr(self, '_last_pseudo_loss') and (self._last_pseudo_loss is not None):
                        pseudo_loss = self._last_pseudo_loss
                        if torch.is_tensor(pseudo_loss) and (not torch.isnan(pseudo_loss)) and (not torch.isinf(pseudo_loss)):
                            train_loss = train_loss + self._mlpr_lambda_u * pseudo_loss

                    meta_loss = self._meta_step_first_order_fixed(
                        student_model=self.model, meta_net=self.meta, meta_opt=self.meta_opt,
                        train_loss=train_loss, val_batch=(v_val, a_val, y_val),
                        lr_inner=self._mlpr_inner_lr
                    )
                    if not hasattr(self, '_meta_losses'): self._meta_losses = []
                    self._meta_losses.append(meta_loss)
                    self.model.train()
        except Exception as e:
            print(f"[Meta Update] {e}")

    def _meta_step_first_order_fixed(self, student_model, meta_net, meta_opt, train_loss, val_batch, lr_inner):
        param_backup = {n: p.clone() for n, p in student_model.named_parameters() if p.requires_grad}
        grads = torch.autograd.grad(
            train_loss, [p for p in student_model.parameters() if p.requires_grad],
            create_graph=True, allow_unused=True
        )
        idx = 0
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                if grads[idx] is not None:
                    param.data = param.data - lr_inner * grads[idx]
                idx += 1
        v_val, a_val, y_val = val_batch
        out_val = student_model(v_val, a_val, return_aux=False)
        logits_val = out_val.get("clip_logits", list(out_val.values())[0]) if isinstance(out_val, dict) else out_val
        val_loss = F.cross_entropy(logits_val, y_val)
        meta_opt.zero_grad(); val_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_net.parameters(), 1.0)
        meta_opt.step()
        for name, param in student_model.named_parameters():
            if name in param_backup: param.data = param_backup[name]
        return val_loss.item()

    # -------------------- train / validate --------------------
    def train(self):
        for epoch in range(1, self.epochs + 1):
            tr = self._train_epoch(epoch)
            va = self._validate(epoch)
            self.scheduler.step()

            # best 以两路中更优的 f1_macro
            f1_for_ckpt = max(va["student"]["f1_macro"], va["teacher"]["f1_macro"])
            who = "student" if f1_for_ckpt == va["student"]["f1_macro"] else "teacher"

            if f1_for_ckpt > getattr(self, "best_f1", -1.0):
                self.best_f1 = f1_for_ckpt
                torch.save({"epoch": epoch, "state_dict": (self.model.state_dict() if who=="student" else self.teacher.state_dict())},
                           self.out_dir / 'checkpoints' / 'best_f1.pth')
                self.no_improve = 0
            else:
                self.no_improve += 1
                if self.use_ssl:
                    self.lambda_u = max(0.3 * self.lambda_u, 0.1)
                    print(f"[Safety] 验证集未提升，下调 lambda_u 至 {self.lambda_u:.2f}")

            torch.save({"epoch": epoch, "state_dict": self.model.state_dict()},
                       self.out_dir / 'checkpoints' / 'latest.pth')

            lr_head = self.opt.param_groups[0]["lr"]; lr_bb = self.opt.param_groups[1]["lr"]
            nan_rate = self.nan_count / max(self.total_steps, 1) * 100
            meta_fail_rate = (self.meta_fail_count / max(self.total_steps // max(self._mlpr_meta_interval,1), 1) * 100) if self.mlpr_enabled else 0

            print(f"[Epoch {epoch}/{self.epochs}] LRs={lr_head:.6f},{lr_bb:.6f} | "
                  f"Train={tr['loss']:.4f} | "
                  f"Val(student) acc={va['student']['acc']:.4f} f1M={va['student']['f1_macro']:.4f} aucM={va['student']['auc_macro']:.4f} | "
                  f"Val(teacher) acc={va['teacher']['acc']:.4f} f1M={va['teacher']['f1_macro']:.4f} aucM={va['teacher']['auc_macro']:.4f} | "
                  f"CKPT={who} | NaN: {self.nan_count}/{self.total_steps} ({nan_rate:.1f}%) | Meta fails: {self.meta_fail_count} ({meta_fail_rate:.1f}%)")

            if self.early_stop_patience > 0 and self.no_improve >= self.early_stop_patience:
                print(f"[EarlyStop] 验证集 {self.early_stop_patience} 个 epoch 无提升，提前停止。")
                return

    def _train_epoch(self, epoch: int):
        if self.grad_explosion_count >= self.max_grad_explosion and self.amp_enabled:
            print(f"[Epoch {epoch}] 梯度爆炸过多，禁用AMP")
            self.amp_enabled = False; self.scaler = AmpGradScaler(self.device_type, enabled=False)
            for pg in self.opt.param_groups: pg['lr'] *= 0.5
        if epoch >= self.amp_disable_epoch and self.amp_enabled:
            print(f"[Epoch {epoch}] 达到预定epoch，禁用AMP以提高稳定性")
            self.amp_enabled = False; self.scaler = AmpGradScaler(self.device_type, enabled=False)
            for pg in self.opt.param_groups: pg['lr'] *= 0.7

        self.current_epoch = epoch
        self.model.train()
        if self.teacher is not None: self.teacher.eval()

        tot = 0.0; nb = 0; npseudo_mass = 0.0
        u_iter = iter(self.loader_u) if self.use_ssl else None
        thr_ssl = self._thr_at(epoch)
        if not hasattr(self, '_meta_losses'): self._meta_losses = []
        step_count = 0
        self._last_pseudo_loss = None

        ramp_epochs = max(self.ssl_warmup + 5, 1)
        lambda_u_eff_ssl  = float(self.lambda_u)         * ramp_up(epoch, ramp_epochs)
        lambda_u_eff_mlpr = float(self._mlpr_lambda_u)   * ramp_up(epoch, ramp_epochs)

        pbar = tqdm(self.loader_l, desc=f"Training Epoch {epoch}")
        last_gate_mean = None; last_delay_mean = None
        self._last_align = 0.0; self._last_prior = 0.0; self._last_edge = 0.0

        for b in pbar:
            if b is None or len(b) < 3: continue
            if isinstance(b, (list, tuple)) and len(b) == 4: v, a, y, _ = b
            else: v, a, y = b
            v = v.to(self.device); a = a.to(self.device)
            y = y.argmax(dim=1).to(self.device) if y.ndim == 2 else y.to(self.device)

            with amp_autocast(self.device_type, enabled=self.amp_enabled, dtype=torch.float16):
                # ------- 监督 -------
                out = self._safe_forward(v, a, use_amp=True)
                if out is None or "clip_logits" not in out:
                    self.nan_count += 1; self.total_steps += 1; self._reset_scaler_if_needed(); continue
                logits = out["clip_logits"]
                sup_loss = self.criterion(logits, y)

                # ------- CAVA -------
                cava_loss = v.new_zeros([])
                if self.cava_enabled:
                    try:
                        with amp_autocast(self.device_type, enabled=False):
                            use_infonce = bool(self.cava_cfg.get("use_infonce", True))
                            lam_align = float(self.cava_cfg.get("lambda_align", 0.05))
                            lam_prior = float(self.cava_cfg.get("lambda_prior", 0.00))
                            lam_edge  = float(self.cava_cfg.get("lambda_edge", 0.01))

                            a_aln = out.get("audio_aligned", out.get("audio_seq"))
                            v_prj = out.get("video_proj",   out.get("video_seq"))
                            g = out.get("causal_gate", None)

                            if isinstance(a_aln, torch.Tensor) and isinstance(v_prj, torch.Tensor):
                                a_aln = F.normalize(a_aln, dim=-1)
                                v_prj = F.normalize(v_prj, dim=-1)

                            if use_infonce:
                                loss_align = info_nce_align(a_aln, v_prj, mask=g, tau=float(self.cava_cfg.get("tau", 0.07)))
                            else:
                                loss_align = corr_diag_align(a_aln, v_prj, mask=g)
                            loss_align = torch.clamp(loss_align, min=0.0, max=10.0)

                            delta = out.get("delay_frames_cont", out.get("delay_frames", None))
                            loss_prior = v.new_zeros([]); loss_edge = v.new_zeros([])
                            if isinstance(delta, torch.Tensor):
                                low = float(out.get("delta_low", self.cava_cfg.get("delta_low_frames", 2.0)))
                                high = float(out.get("delta_high", self.cava_cfg.get("delta_high_frames", 6.0)))
                                if lam_prior > 0:
                                    mu = self.cava_cfg.get("prior_mu", None)
                                    sigma = self.cava_cfg.get("prior_sigma", None)
                                    if (mu is not None) and (sigma is not None):
                                        loss_prior = prior_l2(delta, mu, sigma)
                                if lam_edge > 0:
                                    loss_edge = edge_hinge(delta, low, high,
                                                           margin_ratio=float(self.cava_cfg.get("edge_margin_ratio", 0.25)))
                            cava_loss = lam_align * loss_align + lam_prior * loss_prior + lam_edge * loss_edge
                            self._last_align = float(getattr(loss_align, "item", lambda: loss_align)())
                            self._last_prior = float(getattr(loss_prior, "item", lambda: loss_prior)())
                            self._last_edge  = float(getattr(loss_edge,  "item", lambda: loss_edge)())
                            cg = out.get("causal_gate", None)
                            dl = out.get("delay_frames_cont", out.get("delay_frames", None))
                            if isinstance(cg, torch.Tensor): last_gate_mean = float(cg.float().mean().detach().cpu())
                            if isinstance(dl, torch.Tensor): last_delay_mean = float(dl.float().mean().detach().cpu())
                    except Exception as e:
                        print(f"⚠️ CAVA损失计算异常: {e}")

                loss = sup_loss + cava_loss
                if self.mlpr_enabled:
                    self._last_train_batch = (v.detach(), a.detach(), y.detach())

                # ------- 半监督 -------
                thr_display = thr_ssl
                if self.use_ssl and (u_iter is not None):
                    try:
                        try:
                            bu = next(u_iter)
                        except StopIteration:
                            u_iter = iter(self.loader_u); bu = next(u_iter)
                        if isinstance(bu, (list, tuple)) and len(bu) == 4:
                            vu, au, yu, ids_u = bu
                        else:
                            vu, au, yu = bu; ids_u = None
                        vu = vu.to(self.device); au = au.to(self.device)

                        with torch.no_grad():
                            tout = self.teacher(vu, au, return_aux=False)
                            t_logits = tout["clip_logits"] if isinstance(tout, dict) and "clip_logits" in tout \
                                else (list(tout.values())[0] if isinstance(tout, dict) else tout)
                            t_logits = torch.clamp(t_logits, min=-50, max=50)
                            t_prob = F.softmax(t_logits / self.ssl_temp, dim=1)

                            # 统计 p90 并更新 EMA
                            t_max_all = t_prob.max(dim=1).values.detach().float().cpu().numpy()
                            p90 = float(np.percentile(t_max_all, 90))
                            self._teach_p90_ema = 0.9 * self._teach_p90_ema + 0.1 * p90
                            if (step_count % 100) == 0:
                                print(f"[Diag] teacher_conf: mean={t_max_all.mean():.3f} p90={p90:.3f} thr={thr_ssl:.2f}")

                            # 全局阈值的更保守形态：不超过 0.9*p90_ema
                            thr_cap = max(self._thr_min, 0.9 * self._teach_p90_ema)
                            thr_local = min(self._thr_at(epoch), thr_cap)
                            thr_display = thr_local

                            # 温和启用 DA / 类阈值
                            enable_da   = self._use_dist_align and ((epoch > self.ssl_warmup + 2) or (self._teach_p90_ema > 0.35))
                            enable_cpl  = self._use_cls_threshold and ((epoch > self.ssl_warmup + 2) or (self._teach_p90_ema > 0.35))

                            if enable_da:
                                q = t_prob.mean(dim=0, keepdim=True).clamp(min=1e-8)
                                align = (self._pi.view(1, -1) / q).to(t_prob)
                                t_prob = (t_prob * align).clamp(min=1e-8)
                                t_prob = t_prob / t_prob.sum(dim=1, keepdim=True)

                            t_max, t_idx = t_prob.max(dim=1)

                            if enable_cpl:
                                with torch.no_grad():
                                    for c in range(self.C):
                                        mask_c = (t_idx == c)
                                        if mask_c.any():
                                            mean_c = t_max[mask_c].mean()
                                            self._cls_conf_ema[c] = self._cls_thr_momentum * self._cls_conf_ema[c] + (1 - self._cls_thr_momentum) * mean_c
                                            self._cls_thr[c] = torch.clamp(self._cls_conf_ema[c], min=self._thr_min, max=self.ssl_final_thresh)
                                thr_vec = self._cls_thr.to(self.device)[t_idx]
                                thr_use = torch.minimum(thr_vec, torch.full_like(thr_vec, thr_local))
                            else:
                                thr_use = torch.full_like(t_max, thr_local)

                        # 学生前向
                        sout = self._safe_forward(vu, au, use_amp=True)
                        if sout is None or "clip_logits" not in sout:
                            pass
                        else:
                            s_logits = sout["clip_logits"]

                            if self.mlpr_enabled and (self.meta is not None):
                                stu_feat = None
                                if isinstance(sout, dict):
                                    ftok = sout.get("fusion_token", None)
                                    if ftok is not None:
                                        stu_feat = ftok.mean(dim=tuple(range(1, ftok.dim()))) if ftok.dim() > 2 \
                                            else (ftok.view(ftok.size(0), -1) if ftok.dim() == 1 else ftok)
                                    else:
                                        vtok = sout.get("video_proj", sout.get("video_emb", None))
                                        atok = sout.get("audio_aligned", sout.get("audio_emb", None))
                                        if (vtok is not None) and (atok is not None):
                                            if vtok.dim() > 2: vtok = vtok.mean(dim=tuple(range(1, vtok.dim())))
                                            elif vtok.dim() == 1: vtok = vtok.view(vtok.size(0), -1)
                                            if atok.dim() > 2: atok = atok.mean(dim=tuple(range(1, atok.dim())))
                                            elif atok.dim() == 1: atok = atok.view(atok.size(0), -1)
                                            stu_feat = torch.cat([vtok, atok], dim=-1)

                                hist_mu = hist_std = None; id_list = None
                                if getattr(self, "hist_bank", None) is not None and (ids_u is not None):
                                    id_list = ids_u.cpu().tolist() if torch.is_tensor(ids_u) else ids_u
                                    mu, sd = self.hist_bank.query([int(x) for x in id_list])
                                    hist_mu = mu.to(self.device); hist_std = sd.to(self.device)
                                    if hist_mu.dim() > 1: hist_mu = hist_mu.mean(dim=tuple(range(1, hist_mu.dim())))
                                    if hist_std.dim() > 1: hist_std = hist_std.mean(dim=tuple(range(1, hist_std.dim())))
                                    hist_mu = hist_mu.view(-1, 1); hist_std = hist_std.view(-1, 1)

                                cava_gate_mean = None
                                if isinstance(sout, dict) and ("causal_gate" in sout) and (sout["causal_gate"] is not None):
                                    cg = sout["causal_gate"]
                                    cava_gate_mean = cg.mean(dim=tuple(range(1, cg.dim()))).view(-1, 1) if cg.dim() > 1 else cg.view(-1, 1)

                                feats = build_mlpr_features(
                                    teacher_prob=t_prob,
                                    student_feat=stu_feat,
                                    history_mean=hist_mu,
                                    history_std=hist_std,
                                    cava_gate_mean=cava_gate_mean,
                                    use_prob_vector=self._mlpr_flags["use_prob_vec"]
                                )
                                if not hasattr(self, "_printed_meta_dim_check"):
                                    print(f"[MLPR] feats.shape={tuple(feats.shape)}, meta.input_dim={getattr(self.meta, 'input_dim', None)}")
                                    self._printed_meta_dim_check = True

                                w = self.meta(feats).clone()
                                s_log_prob = F.log_softmax(s_logits, dim=1)
                                t_prob_stable = t_prob.clamp(min=1e-8)
                                kl = F.kl_div(s_log_prob, t_prob_stable, reduction="none").sum(dim=1, keepdim=True)
                                kl = torch.clamp(kl, min=0.0, max=10.0)

                                conf_mask = (t_max.view(-1, 1) > thr_use.view(-1, 1)).float()
                                gate_min = float(self.cava_cfg.get("gate_min", 0.15))
                                if cava_gate_mean is not None:
                                    gate_mask = (cava_gate_mean > gate_min).float()
                                    mask = conf_mask * gate_mask
                                else:
                                    mask = conf_mask

                                w_eff = (w * mask).clamp_(0.0, 1.0)
                                mass = float(w_eff.detach().sum().item())
                                if w_eff.sum() > 0:
                                    pseudo_loss = (w_eff * kl).sum() / (w_eff.sum() + 1e-8)
                                else:
                                    pseudo_loss = v.new_zeros([])

                                self._last_pseudo_loss = pseudo_loss.detach()
                                if self.hist_bank is not None and id_list is not None:
                                    self.hist_bank.update(id_list, kl.squeeze(1).detach())
                                loss = loss + lambda_u_eff_mlpr * pseudo_loss
                                npseudo_mass += mass
                            else:
                                mask = (t_max > thr_use)
                                if mask.any():
                                    pseudo_loss = F.cross_entropy(s_logits[mask], t_idx[mask])
                                    self._last_pseudo_loss = pseudo_loss.detach()
                                    loss = loss + lambda_u_eff_ssl * pseudo_loss
                                    npseudo_mass += float(mask.sum().item())
                                else:
                                    self._last_pseudo_loss = None
                    except Exception as e:
                        print(f"⚠️ 半监督部分异常: {e}")

                # ------- 反向传播 -------
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1; self.total_steps += 1; self._reset_scaler_if_needed(); continue
                self.opt.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    try:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.opt); self.scaler.update()
                    except Exception as e:
                        print(f"⚠️ AMP反向传播错误: {e}")
                        self.opt.zero_grad(set_to_none=True); self._reset_scaler_if_needed(); self.nan_count += 1; self.total_steps += 1; continue
                else:
                    try:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.opt.step()
                    except Exception as e:
                        print(f"⚠️ 反向传播错误: {e}")
                        self.opt.zero_grad(set_to_none=True); self.nan_count += 1; self.total_steps += 1; continue

            # 元学习更新 & EMA
            if self.mlpr_enabled and self.meta is not None:
                if (step_count + 1) % max(self._mlpr_meta_interval, 1) == 0:
                    self._meta_update_step(step_count)
            if self.use_ssl and self.teacher is not None:
                # 传入当前 batch 在 epoch 内的进度，用于 ema_now 计算
                frac = (step_count + 1) / max(1, len(self.loader_l))
                self._ema_update(frac_in_epoch=frac)

            tot += float(loss.detach().item()); nb += 1; step_count += 1; self.total_steps += 1
            pbar.set_postfix(
                loss=f"{tot / max(1, nb):.4f}",
                wΣ=f"{npseudo_mass:.0f}",
                thr=f"{thr_display:.2f}",
                lam=f"{(lambda_u_eff_mlpr if self.mlpr_enabled else lambda_u_eff_ssl):.2f}",
                meta=f"{len(getattr(self, '_meta_losses', []))}",
                Lal=f"{self._last_align:.3f}", Lpr=f"{self._last_prior:.3f}", Led=f"{self._last_edge:.3f}",
                nan=f"{self.nan_count}",
                gate=f"{last_gate_mean:.2f}" if last_gate_mean is not None else "-",
                dly=f"{last_delay_mean:.2f}" if last_delay_mean is not None else "-"
            )
        return {"loss": round(tot / max(1, nb), 4)}

    @torch.no_grad()
    def _validate(self, epoch: int):
        def _eval_model(m):
            m.eval()
            ys, ps = [], []
            for b in DataLoader(self.ds_v, batch_size=self.bs, shuffle=False, num_workers=0,
                                pin_memory=(self.device.type == 'cuda'), drop_last=False, collate_fn=safe_collate_fn):
                if isinstance(b, (list, tuple)) and len(b) == 4: v, a, y, _ = b
                else: v, a, y = b
                v = v.to(self.device); a = a.to(self.device)
                y = y.argmax(dim=1) if y.ndim == 2 else y
                out = m(v, a, return_aux=False)
                logits = out["clip_logits"] if isinstance(out, dict) and "clip_logits" in out else (list(out.values())[0] if isinstance(out, dict) else out)
                prob = F.softmax(logits, dim=1).cpu().numpy()
                ps.append(prob); ys.append(y.cpu().numpy())
            y_true = np.concatenate(ys, 0); y_prob = np.concatenate(ps, 0); y_pred = y_prob.argmax(1)
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            acc = accuracy_score(y_true, y_pred); f1m = f1_score(y_true, y_pred, average="macro")
            aucm = np.nan
            try:
                y_true_oh = np.eye(self.C, dtype=np.float32)[y_true]
                aucm = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
            except Exception:
                pass
            return {"acc": float(acc), "f1_macro": float(f1m), "auc_macro": float(aucm)}

        stu = _eval_model(self.model)
        tea = _eval_model(self.teacher) if (self.teacher is not None) else {"acc":0.0,"f1_macro":0.0,"auc_macro":0.0}
        return {"student": stu, "teacher": tea}

    def _build_sampler(self, ds_l, inv_freq):
        labels = []
        if hasattr(ds_l, "rows") and len(ds_l.rows) > 0:
            for r in ds_l.rows:
                y = r.get("label_idx", None); y = int(y) if (y is not None) else -1; labels.append(y)
        else:
            for i in range(len(ds_l)):
                try:
                    item = ds_l[i]; y = item[2] if isinstance(item, (list, tuple)) else None
                    if torch.is_tensor(y): y = int(y.item()); labels.append(int(y) if y is not None else -1)
                except Exception:
                    labels.append(-1)
        C = len(inv_freq)
        weights = np.zeros(len(labels), dtype=np.float64)
        for i, y in enumerate(labels):
            if 0 <= y < C: weights[i] = float(inv_freq[y])
            else: weights[i] = float(inv_freq.mean()) if C > 0 else 1.0
        w = torch.tensor(weights, dtype=torch.double)
        return WeightedRandomSampler(w, num_samples=len(ds_l), replacement=True)

    def _safe_forward(self, v: torch.Tensor, a: torch.Tensor, use_amp: bool = True):
        try:
            if torch.isnan(v).any() or torch.isinf(v).any():
                v = torch.where(torch.isnan(v) | torch.isinf(v), torch.zeros_like(v), v)
            if torch.isnan(a).any() or torch.isinf(a).any():
                a = torch.where(torch.isnan(a) | torch.isinf(a), torch.zeros_like(a), a)
            current_epoch = getattr(self, 'current_epoch', 1)
            if current_epoch >= self.amp_disable_epoch: use_amp = False
            if use_amp and self.amp_enabled:
                with amp_autocast(self.device_type, enabled=True, dtype=torch.float16):
                    out = self._forward(v, a)
            else:
                v = v.float(); a = a.float()
                with amp_autocast(self.device_type, enabled=False):
                    out = self._forward(v, a)
            return out
        except Exception as e:
            print(f"⚠️ 前向传播异常: {e}")
            return None
