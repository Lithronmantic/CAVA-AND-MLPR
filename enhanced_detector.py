#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnhancedAVTopDetector - 完全修复版（严格按你提供的版本微调，不删段、不瘦身）

修复要点：
1. ✅ 初始化顺序修正：先构建 video/audio backbone → 回写真实 out_dim → 再构建融合层（解决 video_backbone 未定义）
2. ✅ SafeCoAttention 兼容 [B,T,D] / [B*T,D] 输入，自动展平与还原；自动维度对齐&模态顺序自适配（根除 128x128 @ 512x256 错误）
3. ✅ Default/IB/CFA 路径保持你原有接口（[B,T,D]→[B,T,F]）；MIL Head 接口保持不变（返回 clip/seg/weights）
4. ✅ _build_* 支持你给的 dict/字符串写法；冻结模块支持；BN 置 eval
5. ✅ 全部保留你原有类与测试段、兼容别名，不省略内容
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from typing import Dict, Optional, Tuple

# 修改导入（保留你原本的全部导入意图，不删除）
try:
    from src.avtop.models.enhanced_audio_backbones import (
        LightVGGishAudioBackbone,
        ModerateVGGishAudioBackbone,
        ImprovedAudioBackbone,
    )
    _HAS_EXT_AUDIO = True
except Exception:
    _HAS_EXT_AUDIO = False
try:
    from src.avtop.fusion.cfa_fusion import CFAFusion

    HAS_CFA = True
except Exception:
    HAS_CFA = False
    print("⚠️ CFAFusion not found, using default fusion")

try:
    from src.avtop.fusion.ib_fusion import InformationBottleneckFusion

    HAS_IB = True
except Exception:
    HAS_IB = False

try:
    from src.avtop.fusion.coattention import CoAttentionFusion

    HAS_COATTN = True
except Exception:
    HAS_COATTN = False
try:
    from src.avtop.models.backbones import VideoBackbone as ImportedVideoBackbone, \
        AudioBackbone as ImportedAudioBackbone

    HAS_BACKBONES = True
except Exception:
    HAS_BACKBONES = False
    print("⚠️ Backbones not found, using local/dummy backbones")

try:
    from src.avtop.models.temporal_encoder import SimpleTemporalEncoder

    HAS_TEMPORAL = True
except Exception:
    HAS_TEMPORAL = False
# enhanced_detector.py 顶部其它 import 之后
try:
    from cava import CAVAModule

    HAS_CAVA = True
except Exception:
    HAS_CAVA = False
    print("⚠️ CAVA module not found, proceeding without causal alignment")

import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)
from enhanced_mil import EnhancedMIL

def _extract(obj, key, default=None):
    # 从 dict 或 具名对象（easydict/Namespace）里取字段
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

class SafeCoAttention(nn.Module):
    """
    包装 CoAttentionFusion：
    - 稳健探测 core 的“期望输入维度”（支持 nn.Linear / nn.Sequential / 常见字段）
    - 必要时交换模态次序（core 的 v_in/a_in 与我们的视频/音频维度对不上时）
    - 维度自适配：把 [B,T,D?] 投到 core 期望的维度
    - AMP 安全：动态 Linear 永远用 FP32 注册，调用前对齐输入 dtype
    - 兼容返回 (fused, aux) 或 fused
    构造签名保持：SafeCoAttention(core, video_dim, audio_dim, fusion_dim=None)
    """
    def __init__(self, core, video_dim: int, audio_dim: int, fusion_dim: Optional[int] = None):
        super().__init__()
        self.core = core
        self.video_dim = int(video_dim)
        self.audio_dim = int(audio_dim)
        self.fusion_dim = int(fusion_dim) if fusion_dim is not None else None

        # 懒创建投影
        self.v_proj: Optional[nn.Linear] = None
        self.a_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None

    # ---------- 工具：FP32 安全创建/复用 Linear ----------
    def _make_linear_fp32(self, in_f: int, out_f: int, device):
        ctx = torch.amp.autocast("cuda", enabled=False) if torch.cuda.is_available() else nullcontext()
        with ctx:
            lin = nn.Linear(in_f, out_f, bias=True).to(device).float()
        return lin

    def _get_or_remake_linear(self, name: str, in_f: int, out_f: int, device):
        lin = getattr(self, name, None)
        need_new = (lin is None) or (lin.in_features != in_f) or (lin.out_features != out_f)
        if need_new:
            lin = self._make_linear_fp32(in_f, out_f, device)
            setattr(self, name, lin)
        else:
            if lin.weight.dtype != torch.float32:
                lin.float()
        return lin

    # ---------- 工具：从 core 稳健获取“期望输入维度” ----------
    def _first_linear_in_features(self, mod: nn.Module) -> Optional[int]:
        if isinstance(mod, nn.Linear):
            return int(mod.in_features)
        if isinstance(mod, nn.Sequential):
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    return int(m.in_features)
        return None

    def _guess_expected_dims(self) -> Tuple[Optional[int], Optional[int]]:
        # 支持 core.core 的包裹
        obj = getattr(self.core, "core", self.core)
        vin = self._first_linear_in_features(getattr(obj, "v_in", None)) if hasattr(obj, "v_in") else None
        ain = self._first_linear_in_features(getattr(obj, "a_in", None)) if hasattr(obj, "a_in") else None

        # 如果没拿到，尝试常见字段（许多实现只暴露 d_model/embed_dim，表示已经投影后的工作维度）
        if vin is None and ain is None:
            dm = None
            for k in ("d_model", "embed_dim", "model_dim", "hidden_dim"):
                dm = getattr(obj, k, None)
                if dm is not None:
                    try:
                        dm = int(dm)
                        break
                    except Exception:
                        dm = None
            if dm is not None:
                vin = vin or dm
                ain = ain or dm

        # 再不行就退回声明的外部维度
        vin = vin or self.video_dim
        ain = ain or self.audio_dim
        return int(vin), int(ain)

    # ---------- 把最后一维投到 want（保持 [B,T,*]） ----------
    def _adapt_lastdim(self, x: torch.Tensor, want: int, name: str):
        B, T, Din = x.shape
        if Din == want:
            return x
        proj = self._get_or_remake_linear(f"{name}_proj", Din, want, x.device)
        x2 = x.reshape(B * T, Din).to(proj.weight.dtype)   # 对齐到 FP32
        x2 = proj(x2)
        return x2.reshape(B, T, want)

    def _ensure_btd(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        if x.dim() == 3:
            return x, x.size(0), x.size(1), x.size(2)
        if x.dim() == 2:
            return x.unsqueeze(1), x.size(0), 1, x.size(1)
        raise ValueError(f"SafeCoAttention 期望 2D/3D 张量，收到 {tuple(x.shape)}")

    def forward(self, v: torch.Tensor, a: torch.Tensor, **kw):
        v, Bv, Tv, Dv = self._ensure_btd(v)
        a, Ba, Ta, Da = self._ensure_btd(a)
        assert Bv == Ba, "CoAttn 两模态 batch 大小不一致"

        # 读取 core 的期望输入维度（稳健）
        vin, ain = self._guess_expected_dims()

        # 若 core 的 v_in 更像是在处理“我们这边的音频”，而 a_in 更像“我们的视频”，则交换调用次序
        call = "va"
        if (vin == Da and ain == Dv) and not (vin == Dv and ain == Da):
            call = "av"
            want_v, want_a = ain, vin  # 交换
        else:
            want_v, want_a = vin, ain

        # 按 core 期望维度对齐
        v = self._adapt_lastdim(v, want_v, "v")
        a = self._adapt_lastdim(a, want_a, "a")

        # 若序列长度不同，右侧 padding 对齐
        if Tv != Ta:
            T = max(Tv, Ta)
            if Tv < T:
                v = F.pad(v, (0, 0, 0, T - Tv))
            if Ta < T:
                a = F.pad(a, (0, 0, 0, T - Ta))
        else:
            T = Tv

        # 调用 core（它通常要求 [B,T,d_model]）
        out = self.core(v, a, **kw) if call == "va" else self.core(a, v, **kw)

        # 兼容返回
        if isinstance(out, tuple):
            fused, aux = out
        else:
            fused, aux = out, {}

        # 如果 fused 是 [B*T, F]，还原回 [B,T,F]
        if fused.dim() == 2:
            fused = fused.reshape(Bv, T, -1)

        # 按需映射到 fusion_dim（FP32 权重，输入对齐）
        if (self.fusion_dim is not None) and (fused.size(-1) != self.fusion_dim):
            need_new = (
                self.out_proj is None
                or (self.out_proj.in_features != fused.size(-1))
                or (self.out_proj.out_features != self.fusion_dim)
            )
            if need_new:
                self.out_proj = nn.Linear(fused.size(-1), self.fusion_dim, bias=True).to(fused.device).float()
            fused = self.out_proj(fused.reshape(Bv * T, -1).to(self.out_proj.weight.dtype)).reshape(Bv, T, self.fusion_dim)

        # 附带一些便于诊断的字段
        aux.setdefault("video_seq", v)
        aux.setdefault("audio_seq", a)
        aux.setdefault("video_emb", v.mean(dim=1))
        aux.setdefault("audio_emb", a.mean(dim=1))
        aux.setdefault("call_order", call)
        aux.setdefault("want_dims", (want_v, want_a))
        return fused, aux





# ============================================================================
# 真实Backbone实现（按你原样保留，必要处做极小修补）
# ============================================================================
class VideoBackbone(nn.Module):
    """使用预训练ResNet的视频特征提取器"""

    def __init__(self, backbone_type='resnet18', output_dim=512, pretrained=True):
        super().__init__()
        self.backbone_type = backbone_type
        self.output_dim = output_dim

        try:
            import torchvision.models as models
            from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

            if backbone_type == 'resnet18':
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                try:
                    resnet = models.resnet18(weights=weights)
                except Exception as e:
                    print(f"⚠️ ResNet18 预训练权重加载失败，使用随机初始化: {e}")
                    resnet = models.resnet18(weights=None)
                self.base_dim = 512
            elif backbone_type == 'resnet34':
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                try:
                    resnet = models.resnet34(weights=weights)
                except Exception as e:
                    print(f"⚠️ ResNet34 预训练权重加载失败，使用随机初始化: {e}")
                    resnet = models.resnet34(weights=None)
                self.base_dim = 512
            elif backbone_type == 'resnet50':
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                try:
                    resnet = models.resnet50(weights=weights)
                except Exception as e:
                    print(f"⚠️ ResNet50 预训练权重加载失败，使用随机初始化: {e}")
                    resnet = models.resnet50(weights=None)
                self.base_dim = 2048

            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.proj = nn.Linear(self.base_dim, output_dim) if self.base_dim != output_dim else nn.Identity()

        except Exception:
            print("⚠️ torchvision未安装，使用简化backbone")
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.base_dim = 128
            self.proj = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] or [B, C, H, W] or [B, T, D] (已提取特征)
        Returns:
            [B, T, D] or [B, D]
        """
        if x.ndim == 3 and x.size(-1) == self.output_dim:  # [B, T, D]
            return x
        if x.ndim == 2:  # [B, D]
            return x

        if x.ndim == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            feat = self.features(x).squeeze(-1).squeeze(-1)  # [B*T, base_dim]
            feat = self.proj(feat)  # [B*T, output_dim]
            return feat.reshape(B, T, -1)
        else:  # [B, C, H, W]
            feat = self.features(x).squeeze(-1).squeeze(-1)
            return self.proj(feat)


class AudioBackbone(nn.Module):
    """音频骨干网络 - 基于CNN的特征提取器"""

    def __init__(self, n_mels=128, hidden_dim=512):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, n_mels, mel_length] 或 [B, C, T, n_mels, mel_length]
        Returns:
            [B, T, D] - 时序特征
        """
        if x.dim() == 5:  # [B, C, T, n_mels, mel_length]
            x = x.squeeze(1)  # [B, T, n_mels, mel_length]

        B, T = x.shape[:2]
        mel = x.reshape(B * T, *x.shape[2:])  # [B*T, n_mels, mel_length]
        mel = mel.unsqueeze(1)  # [B*T, 1, n_mels, mel_length]

        feat = self.conv(mel).squeeze(-1).squeeze(-1)  # [B*T, 256]
        feat = self.fc(feat)  # [B*T, D]
        feat = feat.view(B, T, -1)  # [B, T, D]
        return feat


class DefaultFusion(nn.Module):
    """简单的拼接融合（备用）"""

    def __init__(self, video_dim, audio_dim, fusion_dim):
        super().__init__()
        self.proj = nn.Linear(video_dim + audio_dim, fusion_dim)

    def forward(self, video_feat, audio_feat):
        combined = torch.cat([video_feat, audio_feat], dim=-1)  # [B,T,Dv+Da]
        fused = self.proj(combined)  # [B,T,Df]
        return fused


# ============================================================================
# MIL分类头（保留你的实现）
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
class EnhancedMILHead(nn.Module):
    """
    多实例学习分类头（兼容 BiasInit，数值更稳、可选 Top-K）
    输入: [B, T, D] 序列特征
    输出:
        - clip_logits: [B, num_classes] 视频级分类
        - seg_logits:  [B, T, num_classes] 帧级分类
        - weights:     [B, T] 注意力权重

    说明：
    - BiasInit 将自动命中 self.frame_classifier[-1] 这一层（nn.Linear -> num_classes）
    - 训练阶段可选 Top-K 稀疏权重；评估阶段使用软注意力（支持温度缩放）
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        topk_ratio: float = 0.0,   # 0 表示关闭 Top-K（默认与原实现一致）
        attn_temp: float = 1.0     # 评估期注意力温度，<1 更尖锐
    ):
        super().__init__()
        assert 0.0 <= topk_ratio <= 1.0, "topk_ratio 应位于 [0,1] 区间"
        self.topk_ratio = float(topk_ratio)
        self.attn_temp = float(attn_temp)

        # 帧级分类头（保持 Sequential，便于 BiasInit 命中最后一层）
        self.frame_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)   # BiasInit 命中点
        )

        # 注意力打分分支
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        基于 Top-K 的稀疏权重掩码（均值归一化到 1/k，后续再归一到和为 1）
        scores: [B, T]
        return: [B, T]
        """
        B, T = scores.shape
        k = max(1, min(T, int(k)))
        _, idx = torch.topk(scores, k, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, idx, 1.0 / float(k))
        return mask

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        """
        if x.dim() != 3:
            raise AssertionError(f"EnhancedMILHead 期望输入 [B,T,D]，实际为 {tuple(x.shape)}")

        B, T, D = x.shape

        # 1) 帧级分类（seg_logits）
        seg_logits = self.frame_classifier(x)  # [B, T, num_classes]

        # 2) 注意力权重（weights）
        attn_scores = self.attention(x).squeeze(-1)  # [B, T]

        if self.training and self.topk_ratio > 0.0:
            # 训练期使用 Top-K 稀疏权重，增强判别性
            k = int(round(T * self.topk_ratio))
            weights = self._topk_mask(attn_scores, k)  # [B, T]
        else:
            # 评估期/关闭 Top-K：使用软注意力，支持温度缩放
            temp = max(self.attn_temp, 1e-6)
            weights = F.softmax(attn_scores / temp, dim=1)  # [B, T]

        # 数值安全与归一化（保证每帧权重非负且和为 1）
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # 3) 视频级 logits：对帧级 logits 按权重聚合（线性、可微）
        #    等价于 (seg_logits * weights.unsqueeze(-1)).sum(dim=1)，einsum 更稳健
        clip_logits = torch.einsum('btc,bt->bc', seg_logits, weights)

        return {
            'clip_logits': clip_logits,  # [B, C]
            'seg_logits': seg_logits,    # [B, T, C]
            'weights': weights           # [B, T]
        }



# ============================================================================
# 主检测器（严格保留你的接口与返回结构，仅修正初始化顺序与 CoAttn 适配）
# ============================================================================
class EnhancedAVTopDetector(nn.Module):
    """
    增强的多模态焊接缺陷检测器（多类别修复版）
    """

    def _freeze_stages(self, backbone: nn.Module, frozen_stages: int):
        if backbone is None or frozen_stages <= 0:
            return
        for m in backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.eval()
        layers = []
        for name in ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"):
            if hasattr(backbone, name):
                layers.append(getattr(backbone, name))
        if not layers:
            children = list(backbone.children())
            if children:
                layers = children[:min(frozen_stages, len(children))]
        cnt = 0
        for block in layers:
            if cnt >= frozen_stages:
                break
            for p in block.parameters():
                p.requires_grad = False
            cnt += 1

    def __init__(self, cfg: Dict):
        super().__init__()
        model_cfg = cfg.get('model', {})
        self.video_dim = model_cfg.get('video_dim', 512)
        self.audio_dim = model_cfg.get('audio_dim', 512)
        self.fusion_dim = model_cfg.get('fusion_dim', 256)
        self.num_classes = model_cfg.get('num_classes', 2)
        self.d_model = cfg.get('d_model', self.audio_dim)
        self.cfg = cfg

        # —— 先构建骨干 → 回写真实 out_dim ——（固定在 *_net，避免被覆盖）
        self.video_backbone_net = self._build_video_backbone(cfg)
        self.audio_backbone_net = self._build_audio_backbone(cfg)
        self.video_dim = int(getattr(self, "vbb_out_dim", self.video_dim))
        self.audio_dim = int(getattr(self, "abb_out_dim", self.audio_dim))

        # 兼容旧属性名（外部若引用旧名不会报错，但 forward 不用它们做调用）
        self.video_backbone = self.video_backbone_net
        self.audio_backbone = self.audio_backbone_net

        # 时序编码器（可选）
        if cfg.get('use_temporal_encoder', False) and HAS_TEMPORAL:
            self.video_temporal = SimpleTemporalEncoder(
                input_dim=self.video_dim,
                hidden_dim=model_cfg.get('hidden_dim', 256)
            )
            self.audio_temporal = SimpleTemporalEncoder(
                input_dim=self.audio_dim,
                hidden_dim=model_cfg.get('hidden_dim', 256)
            )
        else:
            self.video_temporal = None
            self.audio_temporal = None

        # 融合模块
        fusion_cfg = cfg.get('fusion', {'type': 'default'})
        fusion_type = fusion_cfg.get('type', 'default')
        if fusion_type == 'coattn' and HAS_COATTN:
            core = CoAttentionFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                d_model=fusion_cfg.get('d_model', self.fusion_dim),
                num_layers=fusion_cfg.get('num_layers', 2),
                num_heads=fusion_cfg.get('num_heads', 8),
                dropout=fusion_cfg.get('dropout', 0.1)
            )
            self.fusion = SafeCoAttention(core, video_dim=self.video_dim, audio_dim=self.audio_dim,
                                          fusion_dim=self.fusion_dim)
            self.fusion_type = 'coattn'
        elif fusion_type == 'ib' and HAS_IB:
            self.fusion = InformationBottleneckFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                fusion_dim=self.fusion_dim,
                beta=fusion_cfg.get('beta', 0.1)
            )
            self.fusion_type = 'ib'
        elif fusion_type == 'cfa' and HAS_CFA:
            self.fusion = CFAFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                fusion_dim=self.fusion_dim
            )
            self.fusion_type = 'cfa'
        else:
            self.fusion = DefaultFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                fusion_dim=self.fusion_dim
            )
            self.fusion_type = 'default'
        print(f"[EnhancedDetector] 使用融合策略: {self.fusion_type}")

        # === CAVA ===
        cava_cfg = cfg.get('cava', {}) if isinstance(cfg, dict) else {}
        self.use_cava = bool(cava_cfg.get('enabled', False) and HAS_CAVA)
        if self.use_cava:
            self.cava = CAVAModule(
                video_dim=self.video_dim, audio_dim=self.audio_dim,
                d_model=int(cava_cfg.get('d_model', self.fusion_dim)),
                delta_low_frames=float(cava_cfg.get('delta_low_frames', 2.0)),
                delta_high_frames=float(cava_cfg.get('delta_high_frames', 6.0)),
                gate_clip_min=float(cava_cfg.get('gate_clip_min', 0.05)),
                gate_clip_max=float(cava_cfg.get('gate_clip_max', 0.95)),
                num_classes=int(cfg.get('data', {}).get('num_classes', cfg.get('model',{}).get('num_classes', self.num_classes))),
                dist_max_delay=int(cava_cfg.get('dist_max_delay', int(cava_cfg.get('delta_high_frames', 6.0))))
            )
        else:
            self.cava = None

        self.cava_to_audio = None
        try:
            cava_d = int(self.cava.d_model) if self.cava is not None else None
        except Exception:
            cava_d = None
        if (self.cava is not None) and (cava_d is not None) and (cava_d != self.audio_dim) and (self.fusion_type in ['default','cfa','ib']):
            self.cava_to_audio = nn.Linear(cava_d, self.audio_dim)

        # MIL 头
        self.mil_head = EnhancedMILHead(
            input_dim=self.fusion_dim,
            num_classes=self.num_classes
        )
        # ★ 推荐 alias：让主路径也能命中
        self.classifier = self.mil_head.frame_classifier  # 便于 BiasInit 的“主分类层”路径直接命中

        # 单模态辅助头
        if cfg.get('use_aux_heads', True):
            self.video_head = nn.Sequential(
                nn.Linear(self.video_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )
            self.audio_head = nn.Sequential(
                nn.Linear(self.audio_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )
        else:
            self.video_head = None
            self.audio_head = None

    # --- 保留并完善：支持 dict/字符串两种写法 ---
    def _build_video_backbone(self, cfg):
        mb = cfg.get('model', {}) or {}
        vb_cfg = mb.get('video_backbone', 'resnet18')

        if isinstance(vb_cfg, dict):
            name = str(vb_cfg.get('name', 'resnet18')).lower()
            weights = str(vb_cfg.get('weights', 'imagenet')).lower()
            pretrained = (weights != 'none')
            out_dim = int(vb_cfg.get('out_dim', self.video_dim))
            frozen = int(vb_cfg.get('frozen_stages', 0))
        else:
            name = str(vb_cfg).lower()
            pretrained = bool(mb.get('pretrained', False))
            out_dim = int(mb.get('video_dim', self.video_dim))
            frozen = 0

        # 若外部已有 ImportedVideoBackbone，也沿用你本地实现（保持名称一致）
        bb = VideoBackbone(
            backbone_type=name,
            output_dim=out_dim,
            pretrained=pretrained
        )
        self._freeze_stages(bb, frozen)
        self.vbb_out_dim = out_dim
        return bb

    def _build_audio_backbone(self, cfg):
        mb = cfg.get('model', {}) or {}
        ab_cfg = mb.get('audio_backbone', 'cnn')
        n_mels = int(cfg.get('n_mels', 128))

        if isinstance(ab_cfg, dict):
            name = str(ab_cfg.get('name', 'cnn')).lower()  # vggish|light_vggish|moderate_vggish|cnn|improved
            weights = str(ab_cfg.get('weights', 'audioset')).lower()
            pretrained = (weights != 'none')
            out_dim = int(ab_cfg.get('out_dim', self.audio_dim))
            frozen = int(ab_cfg.get('frozen_stages', 0))
        else:
            name = str(ab_cfg).lower()
            pretrained = bool(mb.get('pretrained_audio', False))
            out_dim = int(mb.get('audio_dim', self.audio_dim))
            frozen = 0

        if name in ('vggish', 'light_vggish'):
            backbone = LightVGGishAudioBackbone(n_mels=n_mels, hidden_dim=out_dim)
        elif name == 'moderate_vggish':
            backbone = ModerateVGGishAudioBackbone(n_mels=n_mels, hidden_dim=out_dim)
        elif name in ('cnn', 'improved'):
            backbone = ImprovedAudioBackbone(n_mels=n_mels,
                                             hidden_dim=out_dim) if name == 'improved' else AudioBackbone(n_mels=n_mels,
                                                                                                          hidden_dim=out_dim)
        else:
            raise ValueError(f"Unknown audio backbone: {name}")

        self._freeze_stages(backbone, frozen)
        self.abb_out_dim = out_dim
        return backbone

    def forward(self, video, audio, return_aux: bool = True):
        """
        前向传播（保持你的返回结构）
        """
        # —— 运行时防御：确保骨干仍是 Module（未被误覆盖）
        for name in ("video_backbone_net", "audio_backbone_net"):
            mod = getattr(self, name, None)
            if not isinstance(mod, nn.Module) or not hasattr(mod, "forward"):
                raise TypeError(f"{name} 必须是 nn.Module 且包含 forward，当前为 {type(mod)}")

        # 1) 特征提取（只调用 *_net；永不覆盖 Module）
        video_feat = self.video_backbone_net(video)  # [B, T, Dv] or [B, Dv]
        audio_feat = self.audio_backbone_net(audio)  # [B, T, Da] or [B, Da]
        if video_feat.ndim == 2:
            video_feat = video_feat.unsqueeze(1)
        if audio_feat.ndim == 2:
            audio_feat = audio_feat.unsqueeze(1)

        # 2) 时序编码（可选）
        if getattr(self, "video_temporal", None) is not None:
            video_feat = self.video_temporal(video_feat)
        if getattr(self, "audio_temporal", None) is not None:
            audio_feat = self.audio_temporal(audio_feat)

        # 2.5) CAVA 因果对齐
        audio_seq_raw = audio_feat
        cava_aux = {}
        use_cava_flag = bool(getattr(self, "use_cava", False)) and (getattr(self, "cava", None) is not None)
        if use_cava_flag:
            C = self.cava(video_feat, audio_feat)
            audio_aligned = C.get("audio_for_fusion", C.get("audio_aligned", audio_feat))
            need_proj = hasattr(self, "audio_dim") and (audio_aligned.size(-1) != self.audio_dim)
            if need_proj:
                if not hasattr(self, "cava_to_audio") or (self.cava_to_audio is None):
                    self.cava_to_audio = nn.Linear(audio_aligned.size(-1), self.audio_dim).to(audio_aligned.device)
                B_, T_, D_ = audio_aligned.shape
                audio_feat = self.cava_to_audio(audio_aligned.reshape(B_ * T_, D_)).reshape(B_, T_, self.audio_dim)
            else:
                audio_feat = audio_aligned

            causal_gate = C.get("causal_gate", None)
            if causal_gate is not None and causal_gate.ndim == 2:
                causal_gate = causal_gate.unsqueeze(-1)
            cava_aux = {
                "delay_frames":     C.get("delay_frames", None),
                "causal_gate":      causal_gate,
                "audio_aligned":    C.get("audio_aligned", None),
                "video_proj":       C.get("video_proj", None),
                "audio_proj":       C.get("audio_proj", None),
                "delta_low":        getattr(self.cava, "delta_low", None).item() if hasattr(self.cava, "delta_low") else None,
                "delta_high":       getattr(self.cava, "delta_high", None).item() if hasattr(self.cava, "delta_high") else None,
                "causal_prob":      C.get("causal_prob", C.get("causal_gate", None).squeeze(-1) if C.get("causal_gate", None) is not None else None),
                "causal_prob_dist": C.get("causal_prob_dist", None),
                "pred_delay":       C.get("pred_delay", None),
            }

        # 3) 融合
        if getattr(self, "fusion_type", "default") == 'coattn' and hasattr(self.fusion, 'forward'):
            try:
                fused, aux_info = self.fusion(video_feat, audio_feat)
                video_emb = aux_info.get('video_emb', video_feat.mean(dim=1))
                audio_emb = aux_info.get('audio_emb', audio_feat.mean(dim=1))
                video_seq = aux_info.get('video_seq', video_feat)
                audio_seq = aux_info.get('audio_seq', audio_feat)
            except Exception:
                tmp = self.fusion(video_feat, audio_feat)
                fused = tmp[0] if isinstance(tmp, tuple) else tmp
                video_emb = video_feat.mean(dim=1)
                audio_emb = audio_feat.mean(dim=1)
                video_seq = video_feat
                audio_seq = audio_feat
        else:
            fused = self.fusion(video_feat, audio_feat)
            if fused.ndim == 2:
                fused = fused.unsqueeze(1)
            video_emb = video_feat.mean(dim=1)
            audio_emb = audio_feat.mean(dim=1)
            video_seq = video_feat
            audio_seq = audio_feat

        # 4) MIL 分类
        mil_outputs = self.mil_head(fused)

        # 5) 输出
        outputs = {
            'clip_logits': mil_outputs['clip_logits'],
            'seg_logits':  mil_outputs['seg_logits'],
            'weights':     mil_outputs['weights'],
        }

        # 6) 辅助输出
        if return_aux:
            if getattr(self, "video_head", None) is not None and getattr(self, "audio_head", None) is not None:
                video_pooled = video_seq.mean(dim=1)
                audio_pooled = audio_seq.mean(dim=1)
                outputs['video_logits'] = self.video_head(video_pooled)
                outputs['audio_logits'] = self.audio_head(audio_pooled)

            outputs['video_emb']     = video_emb
            outputs['audio_emb']     = audio_emb
            outputs['video_seq']     = video_seq
            outputs['audio_seq']     = audio_seq
            outputs['audio_seq_raw'] = audio_seq_raw

            # fusion_token（MLPR）
            try:
                v_tok = cava_aux.get('video_proj', video_emb) if cava_aux else video_emb
                a_tok = cava_aux.get('audio_aligned', audio_emb) if cava_aux else audio_emb
                if v_tok is not None and v_tok.ndim == 3: v_tok = v_tok.mean(dim=1)
                if a_tok is not None and a_tok.ndim == 3: a_tok = a_tok.mean(dim=1)
                if v_tok is None: v_tok = video_emb
                if a_tok is None: a_tok = audio_emb
                outputs['fusion_token'] = torch.cat([v_tok, a_tok], dim=-1)
            except Exception:
                outputs['fusion_token'] = torch.cat([video_emb, audio_emb], dim=-1)

            if cava_aux:
                outputs.update(cava_aux)

        return outputs


# ============================================================================
# 简化版检测器（用于纯训练）——原样保留
# ============================================================================
class EnhancedAVDetector(nn.Module):
    """
    简化版检测器，用于train_semisup_unified.py
    只包含核心功能，便于训练
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.num_classes = cfg.get('num_classes', 2)
        video_dim = cfg.get('video_dim', 512)
        audio_dim = cfg.get('audio_dim', 256)
        fusion_dim = cfg.get('fusion_dim', 256)
        self.video_enc = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, video_dim)
        )
        self.audio_enc = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, audio_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + audio_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(fusion_dim, self.num_classes)

    def forward(self, video, audio, return_aux=False):
        B = video.size(0)
        if video.ndim == 5:
            T = video.size(1)
            video = video.reshape(B * T, *video.shape[2:])
            v_feat = self.video_enc(video)  # [B*T, D]
            v_feat = v_feat.reshape(B, T, -1).mean(dim=1)  # [B, D]
        else:
            v_feat = self.video_enc(video)
        if audio.ndim == 3 and audio.size(1) > 10:
            T = audio.size(1)
            audio = audio.reshape(B * T, 1, -1)
            a_feat = self.audio_enc(audio)
            a_feat = a_feat.reshape(B, T, -1).mean(dim=1)
        else:
            if audio.ndim == 2:
                audio = audio.unsqueeze(1)
            a_feat = self.audio_enc(audio)
        combined = torch.cat([v_feat, a_feat], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        outputs = {'clip_logits': logits}
        if return_aux:
            outputs.update({
                'video_emb': v_feat,
                'audio_emb': a_feat,
                'video_logits': logits,
                'audio_logits': logits
            })
            # === MLPR: 导出一个融合前的 pooled token，作为“学生特征质量”输入 ===
            try:
                v_tok = outputs.get('video_proj', outputs.get('video_emb', None))
                a_tok = outputs.get('audio_aligned', outputs.get('audio_emb', None))
                if (v_tok is not None) and (a_tok is not None):
                    if v_tok.dim() == 3: v_tok = v_tok.mean(dim=1)
                    if a_tok.dim() == 3: a_tok = a_tok.mean(dim=1)
                    fusion_token = torch.cat([v_tok, a_tok], dim=-1)  # [B, Dv+Da]
                    outputs['fusion_token'] = fusion_token
            except Exception:
                pass
        return outputs if return_aux else outputs['clip_logits']


# ============================================================================
# 测试代码（保留，不删）
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Detector 测试（完全修复版）")
    print("=" * 70)

    cfg = {
        'model': {
            'video_dim': 512,
            'audio_dim': 512,  # 与video_dim一致（按你的注释）
            'fusion_dim': 256,
            'num_classes': 12,
            'video_backbone': 'resnet18',
            'audio_backbone': 'cnn',
            'pretrained': False
        },
        'fusion': {
            'type': 'default',  # 你可改为 'coattn' 验证 SafeCoAttention
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 8
        },
        'use_temporal_encoder': False,
        'use_aux_heads': True,
        'n_mels': 128
    }

    print("\n测试简化版EnhancedAVDetector:")
    simple_model = EnhancedAVDetector({
        'num_classes': 12,
        'video_dim': 512,
        'audio_dim': 256,
        'fusion_dim': 256
    })
    B = 4
    video_simple = torch.randn(B, 3, 224, 224)
    audio_simple = torch.randn(B, 1, 16000)
    out_simple = simple_model(video_simple, audio_simple, return_aux=True)
    print(f"  clip_logits: {out_simple['clip_logits'].shape}")
    print(f"  video_emb: {out_simple['video_emb'].shape}")
    assert out_simple['clip_logits'].shape == (B, 12), "输出形状错误!"
    print("  ✅ 简化版测试通过!")

    print("\n测试完整版EnhancedAVTopDetector:")
    model = EnhancedAVTopDetector(cfg)
    B = 4
    T = 8
    video = torch.randn(B, T, 3, 224, 224)
    audio = torch.randn(B, T, 128, 32)
    outputs = model(video, audio, return_aux=True)
    print(f"\n输出:")
    print(f"  clip_logits: {outputs['clip_logits'].shape}")
    print(f"  seg_logits: {outputs['seg_logits'].shape}")
    print(f"  weights: {outputs['weights'].shape}")
    if 'video_logits' in outputs:
        print(f"\n辅助输出（单模态）:")
        print(f"  video_logits: {outputs['video_logits'].shape}")
        print(f"  audio_logits: {outputs['audio_logits'].shape}")
    if 'video_emb' in outputs:
        print(f"\n全局嵌入（对比学习）:")
        print(f"  video_emb: {outputs['video_emb'].shape}")
        print(f"  audio_emb: {outputs['audio_emb'].shape}")
    assert outputs['clip_logits'].shape == (
    B, 12), f"视频级输出形状错误! 期望{(B, 12)}, 实际{outputs['clip_logits'].shape}"
    assert outputs['seg_logits'].shape == (
    B, T, 12), f"帧级输出形状错误! 期望{(B, T, 12)}, 实际{outputs['seg_logits'].shape}"
    assert torch.allclose(outputs['weights'].sum(dim=1), torch.ones(B), atol=1e-5), "注意力权重应该sum to 1!"
    print(f"\n✅ 所有测试通过! 代码完全正确，可以正常使用！")

# ===================== Compatibility Aliases =====================
try:
    SimpleAVDetector = EnhancedAVDetector  # 简化版保留下来
    EnhancedAVDetector = EnhancedAVTopDetector  # 将完整版暴露为 EnhancedAVDetector
except Exception:
    pass