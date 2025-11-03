#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单样本推理可视化脚本 - 深入展示模型推理过程

功能：
1. 逐帧视频/音频特征可视化
2. CAVA对齐过程动态展示
3. 注意力机制可视化
4. 多模态融合过程
5. 预测置信度演化
6. 可解释性分析

使用方法：
    python inference_visualize.py \
        --checkpoint runs/fixed_exp/checkpoints/best_f1.pth \
        --config selfsup_sota.yaml \
        --video path/to/video.mp4 \
        --audio path/to/audio.wav \
        --output ./inference_vis \
        [--sample_idx 0]  # 或从数据集选择
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import warnings

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches

# 音视频处理
try:
    import cv2
except ImportError:
    print("⚠️  OpenCV未安装，视频可视化可能受限")
    cv2 = None

try:
    import librosa
    import librosa.display
except ImportError:
    print("⚠️  librosa未安装，音频可视化可能受限")
    librosa = None

# 导入模型
from enhanced_detector import EnhancedAVTopDetector
from dataset import AVFromCSV

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


# 配置Windows/macOS/Linux可用的中文字体（并尽量避免缺字告警）
def setup_chinese_font():
    """配置中文字体，支持Windows/macOS/Linux"""
    import platform
    import matplotlib.font_manager as fm

    system = platform.system()
    tried_paths = []

    def try_add_font(path_list):
        for p in path_list:
            if os.path.exists(p):
                try:
                    fm.fontManager.addfont(p)
                    prop = fm.FontProperties(fname=p)
                    plt.rcParams['font.sans-serif'] = [prop.get_name()]
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"✓ 使用字体: {prop.get_name()} @ {p}")
                    return True
                except Exception as e:
                    tried_paths.append((p, str(e)))
        return False

    ok = False
    if system == 'Windows':
        win_fonts = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
        ok = try_add_font([
            os.path.join(win_fonts, 'msyh.ttc'),       # 微软雅黑
            os.path.join(win_fonts, 'msyh.ttf'),
            os.path.join(win_fonts, 'simhei.ttf'),     # 黑体
            os.path.join(win_fonts, 'simsun.ttc'),     # 宋体
            os.path.join(win_fonts, 'msyhbd.ttc'),
        ])
    elif system == 'Darwin':
        ok = try_add_font([
            '/System/Library/Fonts/PingFang.ttc',                # 苹方
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
        ])
    else:
        ok = try_add_font([
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',    # 文泉驿微米黑
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',   # 兜底（不含CJK全量）
        ])

    if not ok:
        print("⚠️  未找到可用中文字体，可能出现缺字警告。可在系统安装 Noto Sans CJK / 微软雅黑 后重试。")

    # 如需静默“Glyph missing”提示，可放开下一行（不影响图像内容）
    # warnings.filterwarnings("ignore", message="Glyph .* missing from current font", category=UserWarning)

setup_chinese_font()


class InferenceVisualizer:
    """单样本推理可视化器"""

    def __init__(
            self,
            model: nn.Module,
            class_names: List[str],
            device: torch.device,
            output_dir: str
    ):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / 'frames').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        (self.output_dir / 'attention').mkdir(exist_ok=True)
        (self.output_dir / 'cava').mkdir(exist_ok=True)
        (self.output_dir / 'fusion').mkdir(exist_ok=True)

        print(f"📁 输出目录: {self.output_dir}")

    @torch.no_grad()
    def visualize_sample(
            self,
            video: torch.Tensor,
            audio: torch.Tensor,
            label: Optional[int] = None,
            sample_name: str = "sample"
    ):
        """完整可视化一个样本"""
        print("\n" + "=" * 60)
        print(f"🎬 开始可视化样本: {sample_name}")
        print("=" * 60)

        self.model.eval()

        # 确保batch维度
        if video.dim() == 4:  # [T,C,H,W]
            video = video.unsqueeze(0)  # [1,T,C,H,W]
        if audio.dim() == 3:  # [T,M,F]
            audio = audio.unsqueeze(0)  # [1,T,M,F]

        video = video.to(self.device)
        audio = audio.to(self.device)

        # 前向传播
        outputs = self.model(video, audio, return_aux=True)

        # 提取预测
        if isinstance(outputs, dict):
            logits = outputs.get('clip_logits', list(outputs.values())[0])
        else:
            logits = outputs

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = logits.argmax(dim=1).item()

        print(f"✅ 预测: {self.class_names[pred]} (置信度: {probs[pred]:.3f})")
        if label is not None:
            print(f"   真实标签: {self.class_names[label]}")
            print(f"   {'✓ 正确' if pred == label else '✗ 错误'}")

        # 1. 输入数据可视化
        self._visualize_input(video[0], audio[0], sample_name)

        # 2. 逐帧特征演化
        self._visualize_temporal_features(video[0], audio[0], outputs, sample_name)

        # 3. CAVA对齐可视化
        if isinstance(outputs, dict):
            self._visualize_cava_alignment(outputs, sample_name)

        # 4. 注意力图
        self._visualize_attention_maps(video[0], audio[0], outputs, sample_name)

        # 5. 融合过程
        self._visualize_fusion_process(outputs, sample_name)

        # 6. 预测分析
        self._visualize_prediction(probs, pred, label, sample_name)

        # 7. 生成总结图
        self._create_summary_figure(
            video[0], audio[0], outputs, probs, pred, label, sample_name
        )

        print(f"💾 可视化完成: {self.output_dir}")

    def _visualize_input(self, video: torch.Tensor, audio: torch.Tensor, name: str):
        """可视化输入数据"""
        print("\n📊 1. 输入数据可视化...")

        video_np = video.cpu().numpy()  # [T, 3, H, W]
        audio_np = audio.cpu().numpy()  # [T, M, F]

        T_v = video_np.shape[0]

        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(3, T_v, figure=fig)

        # 视频帧
        for t in range(T_v):
            ax = fig.add_subplot(gs[0, t])
            frame = video_np[t].transpose(1, 2, 0)  # [H,W,3]
            frame = np.clip(frame, 0, 1)
            ax.imshow(frame)
            ax.set_title(f'Frame {t + 1}', fontsize=10)
            ax.axis('off')

        # 音频光谱图（每帧）
        for t in range(T_v):
            ax = fig.add_subplot(gs[1, t])
            spec = audio_np[t]  # [M, F]
            im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Audio {t + 1}', fontsize=10)
            ax.set_ylabel('Mel bins' if t == 0 else '')
            ax.set_xlabel('Frames')
            if t == T_v - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 音频波形（全局）
        ax = fig.add_subplot(gs[2, :])
        audio_mean = audio_np.mean(axis=1).flatten()  # 展平所有音频
        ax.plot(audio_mean, linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform (aggregated)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'输入数据: {name}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frames' / f'{name}_input.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 输入数据已保存")

    def _visualize_temporal_features(
            self,
            video: torch.Tensor,
            audio: torch.Tensor,
            outputs: Dict,
            name: str
    ):
        """逐帧特征演化"""
        print("\n📈 2. 时序特征演化...")

        if not isinstance(outputs, dict):
            print("  ⚠️  输出不是字典，跳过时序特征")
            return

        # 提取时序特征
        v_feat = outputs.get('video_proj', outputs.get('video_emb'))
        a_feat = outputs.get('audio_aligned', outputs.get('audio_emb'))

        if v_feat is None or a_feat is None:
            print("  ⚠️  缺少时序特征")
            return

        v_feat = v_feat[0].cpu().numpy()  # [T, D]
        a_feat = a_feat[0].cpu().numpy()  # [T, D]

        T = min(v_feat.shape[0], a_feat.shape[0])

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # 视频特征热图
        im1 = axes[0, 0].imshow(v_feat[:T].T, aspect='auto', cmap='viridis')
        axes[0, 0].set_xlabel('Time step')
        axes[0, 0].set_ylabel('Feature dim')
        axes[0, 0].set_title('Video Features', fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        # 音频特征热图
        im2 = axes[0, 1].imshow(a_feat[:T].T, aspect='auto', cmap='viridis')
        axes[0, 1].set_xlabel('Time step')
        axes[0, 1].set_ylabel('Feature dim')
        axes[0, 1].set_title('Audio Features', fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        # 特征范数演化
        v_norm = np.linalg.norm(v_feat[:T], axis=1)
        a_norm = np.linalg.norm(a_feat[:T], axis=1)
        axes[1, 0].plot(range(T), v_norm, 'o-', label='Video', linewidth=2)
        axes[1, 0].plot(range(T), a_norm, 's-', label='Audio', linewidth=2)
        axes[1, 0].set_xlabel('Time step')
        axes[1, 0].set_ylabel('L2 Norm')
        axes[1, 0].set_title('Feature Magnitude', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 模态相似度演化
        v_norm_feat = v_feat[:T] / (np.linalg.norm(v_feat[:T], axis=1, keepdims=True) + 1e-8)
        a_norm_feat = a_feat[:T] / (np.linalg.norm(a_feat[:T], axis=1, keepdims=True) + 1e-8)
        similarity = np.sum(v_norm_feat * a_norm_feat, axis=1)

        axes[1, 1].plot(range(T), similarity, 'o-', linewidth=2)
        axes[1, 1].axhline(similarity.mean(), linestyle='--',
                           label=f'Mean={similarity.mean():.3f}')
        axes[1, 1].set_xlabel('Time step')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].set_title('Modality Similarity Over Time', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # PCA降维可视化
        from sklearn.decomposition import PCA
        combined = np.concatenate([v_feat[:T], a_feat[:T]], axis=0)
        if combined.shape[1] >= 2:
            pca = PCA(n_components=2)
            combined_2d = pca.fit_transform(combined)

            v_2d = combined_2d[:T]
            a_2d = combined_2d[T:]

            axes[2, 0].scatter(v_2d[:, 0], v_2d[:, 1], c=range(T),
                               cmap='Reds', s=100, alpha=0.7, label='Video')
            axes[2, 0].scatter(a_2d[:, 0], a_2d[:, 1], c=range(T),
                               cmap='Blues', s=100, alpha=0.7, marker='s', label='Audio')

            # 连线显示时序
            axes[2, 0].plot(v_2d[:, 0], v_2d[:, 1], 'r-', alpha=0.3, linewidth=1)
            axes[2, 0].plot(a_2d[:, 0], a_2d[:, 1], 'b-', alpha=0.3, linewidth=1)

            axes[2, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[2, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            axes[2, 0].set_title('Feature Trajectory (PCA)', fontweight='bold')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

        # 特征统计
        axes[2, 1].axis('off')
        stats_text = f"""
特征统计 (T={T}):

视频特征:
  - 维度: {v_feat.shape[1]}
  - 均值: {v_feat.mean():.3f}
  - 标准差: {v_feat.std():.3f}
  - 范围: [{v_feat.min():.3f}, {v_feat.max():.3f}]

音频特征:
  - 维度: {a_feat.shape[1]}
  - 均值: {a_feat.mean():.3f}
  - 标准差: {a_feat.std():.3f}
  - 范围: [{a_feat.min():.3f}, {a_feat.max():.3f}]

模态相似度:
  - 均值: {similarity.mean():.3f}
  - 标准差: {similarity.std():.3f}
  - 范围: [{similarity.min():.3f}, {similarity.max():.3f}]
        """
        axes[2, 1].text(0.1, 0.5, stats_text, fontsize=10,
                        verticalalignment='center', family='monospace')

        plt.suptitle(f'时序特征演化: {name}', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'features' / f'{name}_temporal.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 时序特征已保存")

    def _visualize_cava_alignment(self, outputs: Dict, name: str):
        """CAVA对齐可视化"""
        print("\n🎯 3. CAVA对齐过程...")

        gate = outputs.get('causal_gate')
        delay = outputs.get('delay_frames')
        v_proj = outputs.get('video_proj')
        a_align = outputs.get('audio_aligned')

        if gate is None:
            print("  ⚠️  CAVA门控数据不可用")
            return

        gate = gate[0].cpu().numpy()  # [T] or [T, D]
        if delay is not None:
            delay = delay[0].cpu().item() if delay.dim() == 1 else delay[0].cpu().numpy()

        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 3, figure=fig)

        # 门控值演化
        ax1 = fig.add_subplot(gs[0, :2])
        if gate.ndim == 1:
            gate_plot = gate
        else:
            gate_plot = gate.mean(axis=1) if gate.ndim > 1 else gate

        T = len(gate_plot)
        ax1.plot(range(T), gate_plot, 'o-', linewidth=2, markersize=8)
        ax1.fill_between(range(T), 0, gate_plot, alpha=0.3)
        ax1.axhline(gate_plot.mean(), linestyle='--', linewidth=2,
                    label=f'Mean={gate_plot.mean():.3f}')
        ax1.set_xlabel('Time step', fontsize=11)
        ax1.set_ylabel('Gate value', fontsize=11)
        ax1.set_title('CAVA Causal Gate Evolution', fontweight='bold', fontsize=12)
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 延迟信息
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        if delay is not None:
            delay_val = delay if isinstance(delay, (int, float)) else float(np.mean(delay))
            delay_text = f"""
延迟估计:

值: {delay_val:.2f} 帧

含义:
音频相对视频
延迟约 {delay_val:.1f} 帧

门控统计:
均值: {gate_plot.mean():.3f}
最大: {gate_plot.max():.3f}
最小: {gate_plot.min():.3f}
标准差: {gate_plot.std():.3f}
            """
            ax2.text(0.1, 0.5, delay_text, fontsize=10,
                     verticalalignment='center', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 对齐前后对比
        if v_proj is not None and a_align is not None:
            v_proj = v_proj[0].cpu().numpy()  # [T, D]
            a_align = a_align[0].cpu().numpy()

            ax3 = fig.add_subplot(gs[1, :])

            # 计算逐帧相似度
            T_min = min(v_proj.shape[0], a_align.shape[0])
            v_norm = v_proj[:T_min] / (np.linalg.norm(v_proj[:T_min], axis=1, keepdims=True) + 1e-8)
            a_norm = a_align[:T_min] / (np.linalg.norm(a_align[:T_min], axis=1, keepdims=True) + 1e-8)
            sim = np.sum(v_norm * a_norm, axis=1)

            ax3.plot(range(T_min), sim, 'o-', linewidth=2, label='Alignment similarity')
            ax3.fill_between(range(T_min), 0, sim, alpha=0.3)
            ax3.axhline(sim.mean(), linestyle='--', linewidth=2,
                        label=f'Mean={sim.mean():.3f}')
            ax3.set_xlabel('Time step', fontsize=11)
            ax3.set_ylabel('Cosine Similarity', fontsize=11)
            ax3.set_title('Video-Audio Alignment Quality', fontweight='bold', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 相关矩阵
            ax4 = fig.add_subplot(gs[2, 0])
            corr = np.corrcoef(v_proj[:T_min].T, a_align[:T_min].T)[:v_proj.shape[1], v_proj.shape[1]:]
            im = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax4.set_xlabel('Audio features')
            ax4.set_ylabel('Video features')
            ax4.set_title('Cross-modal Correlation', fontweight='bold')
            plt.colorbar(im, ax=ax4, fraction=0.046)

            # 特征距离矩阵
            ax5 = fig.add_subplot(gs[2, 1])
            from scipy.spatial.distance import cdist
            dist = cdist(v_proj[:T_min], a_align[:T_min], metric='euclidean')
            im2 = ax5.imshow(dist, cmap='YlOrRd', aspect='auto')
            ax5.set_xlabel('Audio time step')
            ax5.set_ylabel('Video time step')
            ax5.set_title('Temporal Distance Matrix', fontweight='bold')
            plt.colorbar(im2, ax=ax5, fraction=0.046)

            # 对齐有效性指标
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            alignment_text = f"""
对齐质量评估:

相似度:
  均值: {sim.mean():.3f}
  最大: {sim.max():.3f}
  最小: {sim.min():.3f}

相关性:
  均值: {corr.mean():.3f}
  最大: {corr.max():.3f}

距离:
  均值: {dist.mean():.2f}
  最小: {dist.min():.2f}

评估: {'✓ 良好' if sim.mean() > 0.5 else '⚠ 一般' if sim.mean() > 0.3 else '✗ 较差'}
            """
            ax6.text(0.1, 0.5, alignment_text, fontsize=10,
                     verticalalignment='center', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgreen' if sim.mean() > 0.5 else 'lightyellow',
                               alpha=0.5))

        plt.suptitle(f'CAVA对齐分析: {name}', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cava' / f'{name}_alignment.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ CAVA对齐已保存")

    def _visualize_attention_maps(
            self,
            video: torch.Tensor,
            audio: torch.Tensor,
            outputs: Dict,
            name: str
    ):
        """注意力图可视化"""
        print("\n🔍 4. 注意力机制...")

        if not isinstance(outputs, dict):
            print("  ⚠️  无法提取注意力信息")
            return

        v_feat = outputs.get('video_proj', outputs.get('video_emb'))
        a_feat = outputs.get('audio_aligned', outputs.get('audio_emb'))

        if v_feat is None or a_feat is None:
            print("  ⚠️  缺少特征用于注意力计算")
            return

        v_feat = v_feat[0].cpu().numpy()  # [T, D]
        a_feat = a_feat[0].cpu().numpy()

        T = min(v_feat.shape[0], a_feat.shape[0])

        # 计算交叉注意力（基于相似度）
        v_norm = v_feat[:T] / (np.linalg.norm(v_feat[:T], axis=1, keepdims=True) + 1e-8)
        a_norm = a_feat[:T] / (np.linalg.norm(a_feat[:T], axis=1, keepdims=True) + 1e-8)
        attention = np.dot(v_norm, a_norm.T)  # [T, T]

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 交叉注意力矩阵
        im1 = axes[0, 0].imshow(attention, cmap='YlOrRd', aspect='auto')
        axes[0, 0].set_xlabel('Audio time step')
        axes[0, 0].set_ylabel('Video time step')
        axes[0, 0].set_title('Cross-modal Attention Matrix', fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        # 对角线注意力（时序对齐）
        diag_attention = np.diag(attention)
        axes[0, 1].plot(range(T), diag_attention, 'o-', linewidth=2)
        axes[0, 1].fill_between(range(T), 0, diag_attention, alpha=0.3)
        axes[0, 1].set_xlabel('Time step')
        axes[0, 1].set_ylabel('Attention weight')
        axes[0, 1].set_title('Temporal Alignment Attention', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 视频自注意力
        v_self_attn = np.dot(v_norm, v_norm.T)
        im2 = axes[1, 0].imshow(v_self_attn, cmap='Blues', aspect='auto')
        axes[1, 0].set_xlabel('Video time step')
        axes[1, 0].set_ylabel('Video time step')
        axes[1, 0].set_title('Video Self-Attention', fontweight='bold')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

        # 音频自注意力
        a_self_attn = np.dot(a_norm, a_norm.T)
        im3 = axes[1, 1].imshow(a_self_attn, cmap='Greens', aspect='auto')
        axes[1, 1].set_xlabel('Audio time step')
        axes[1, 1].set_ylabel('Audio time step')
        axes[1, 1].set_title('Audio Self-Attention', fontweight='bold')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

        plt.suptitle(f'注意力分析: {name}', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention' / f'{name}_attention.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 创建注意力热图的动画（可选）
        self._create_attention_animation(attention, name)

        print("  ✓ 注意力图已保存")

    def _create_attention_animation(self, attention: np.ndarray, name: str):
        """创建注意力演化动画"""
        try:
            T = attention.shape[0]

            fig, ax = plt.subplots(figsize=(10, 8))

            def update(frame):
                ax.clear()
                # 显示该时间步的注意力分布
                ax.bar(range(T), attention[frame], alpha=0.7)
                ax.set_xlabel('Attends to time step')
                ax.set_ylabel('Attention weight')
                ax.set_title(f'Attention at time step {frame}', fontweight='bold')
                ax.set_ylim([0, max(1e-8, attention.max() * 1.1)])
                ax.grid(axis='y', alpha=0.3)

            anim = FuncAnimation(fig, update, frames=T, interval=500)
            anim.save(self.output_dir / 'attention' / f'{name}_attention_anim.gif',
                      writer=PillowWriter(fps=2))
            plt.close()

            print("  ✓ 注意力动画已保存")
        except Exception as e:
            print(f"  ⚠️  注意力动画创建失败: {e}")

    def _visualize_fusion_process(self, outputs: Dict, name: str):
        """融合过程可视化"""
        print("\n🔀 5. 多模态融合过程...")

        if not isinstance(outputs, dict):
            print("  ⚠️  无法分析融合过程")
            return

        v_feat = outputs.get('video_proj', outputs.get('video_emb'))
        a_feat = outputs.get('audio_aligned', outputs.get('audio_emb'))
        f_feat = outputs.get('fusion_token', outputs.get('fusion_out'))

        if f_feat is None:
            print("  ⚠️  无融合特征")
            return

        # 提取为 numpy，并将三路都降成 1D（对时序做均值）
        def to_1d(x):
            if x is None:
                return None
            arr = x[0].detach().cpu().numpy()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return arr

        v_feat = to_1d(v_feat)
        a_feat = to_1d(a_feat)
        f_feat = to_1d(f_feat)

        # 统一对齐到最小维度（既用于相似度，也用于可视化与 PCA）
        avail = [x for x in [v_feat, a_feat, f_feat] if x is not None]
        dims = [len(x) for x in avail]
        target_dim = min(dims)

        def align_feature(feat, target):
            if feat is None:
                return None
            if len(feat) > target:
                return feat[:target]
            elif len(feat) < target:
                return np.pad(feat, (0, target - len(feat)), mode='constant')
            else:
                return feat

        v_feat_aligned = align_feature(v_feat, target_dim)
        a_feat_aligned = align_feature(a_feat, target_dim)
        f_feat_aligned = align_feature(f_feat, target_dim)

        if (v_feat is not None and a_feat is not None) and (
            len(v_feat) != len(a_feat) or len(v_feat) != len(f_feat)
        ):
            print(f"  ⚠️  特征维度不一致: Video={len(v_feat)}, Audio={len(a_feat)}, Fusion={len(f_feat)}")
            print(f"     已对齐到 {target_dim} 维进行分析")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 仅当三路都可用时，绘制完整分析
        if (v_feat_aligned is not None) and (a_feat_aligned is not None):
            # 直方图（用对齐后的向量，保证统计可比）
            axes[0, 0].hist([v_feat_aligned, a_feat_aligned, f_feat_aligned], bins=30,
                            label=['Video', 'Audio', 'Fusion'], alpha=0.6)
            axes[0, 0].set_xlabel('Feature value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Feature Distribution', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 范数对比（对齐后）
            feature_names = ['Video', 'Audio', 'Fusion']
            norms = [np.linalg.norm(v_feat_aligned),
                     np.linalg.norm(a_feat_aligned),
                     np.linalg.norm(f_feat_aligned)]
            axes[0, 1].bar(feature_names, norms, alpha=0.7)
            axes[0, 1].set_ylabel('L2 Norm')
            axes[0, 1].set_title('Feature Magnitude', fontweight='bold')
            axes[0, 1].grid(axis='y', alpha=0.3)

            # 相似度分析（余弦）
            from scipy.spatial.distance import cosine
            try:
                v_a_sim = 1 - cosine(v_feat_aligned, a_feat_aligned)
                v_f_sim = 1 - cosine(v_feat_aligned, f_feat_aligned)
                a_f_sim = 1 - cosine(a_feat_aligned, f_feat_aligned)
            except Exception:
                # 退化为归一化点积
                def nz_norm(x):
                    s = np.linalg.norm(x) + 1e-8
                    return x / s
                v_a_sim = float(np.dot(nz_norm(v_feat_aligned), nz_norm(a_feat_aligned)))
                v_f_sim = float(np.dot(nz_norm(v_feat_aligned), nz_norm(f_feat_aligned)))
                a_f_sim = float(np.dot(nz_norm(a_feat_aligned), nz_norm(f_feat_aligned)))

            sim_matrix = np.array([[1, v_a_sim, v_f_sim],
                                   [v_a_sim, 1, a_f_sim],
                                   [v_f_sim, a_f_sim, 1]])

            im = axes[0, 2].imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
            axes[0, 2].set_xticks([0, 1, 2])
            axes[0, 2].set_yticks([0, 1, 2])
            axes[0, 2].set_xticklabels(feature_names)
            axes[0, 2].set_yticklabels(feature_names)
            axes[0, 2].set_title('Inter-modality Similarity', fontweight='bold')
            for i in range(3):
                for j in range(3):
                    axes[0, 2].text(j, i, f'{sim_matrix[i, j]:.2f}',
                                    ha="center", va="center", fontsize=12)
            plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

            # PCA（**使用对齐后的向量**，避免维度不一致导致的 np.stack 错误）
            from sklearn.decomposition import PCA
            combined = np.stack([v_feat_aligned, a_feat_aligned, f_feat_aligned])  # (3, D)
            # n_components 不能超过样本数或维度
            n_comp = 2 if combined.shape[1] >= 2 else 1
            if n_comp >= 1:
                pca = PCA(n_components=n_comp)
                combined_2d = pca.fit_transform(combined)  # (3, n_comp)

                axes[1, 0].scatter(combined_2d[0, 0], combined_2d[0, 1] if n_comp > 1 else 0.0,
                                   s=200, marker='o', label='Video', alpha=0.7)
                axes[1, 0].scatter(combined_2d[1, 0], combined_2d[1, 1] if n_comp > 1 else 0.0,
                                   s=200, marker='s', label='Audio', alpha=0.7)
                axes[1, 0].scatter(combined_2d[2, 0], combined_2d[2, 1] if n_comp > 1 else 0.0,
                                   s=200, marker='*', label='Fusion', alpha=0.7)

                axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})' if n_comp > 1 else 'PC1')
                if n_comp > 1:
                    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                else:
                    axes[1, 0].set_ylabel('PC2 (NA)')
                axes[1, 0].set_title('Fusion in PCA Space', fontweight='bold')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # 融合权重估计（基于相似度）
            total_sim = (v_f_sim + a_f_sim)
            v_weight = v_f_sim / total_sim if total_sim > 0 else 0.5
            a_weight = a_f_sim / total_sim if total_sim > 0 else 0.5

            axes[1, 1].pie([v_weight, a_weight], labels=['Video', 'Audio'],
                           autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Estimated Modality Weights', fontweight='bold')

            # 统计信息（使用对齐后的特征，确保可比）
            axes[1, 2].axis('off')
            fusion_text = f"""
融合统计:

特征维度(对齐后):
  Video: {len(v_feat_aligned)}
  Audio: {len(a_feat_aligned)}
  Fusion: {len(f_feat_aligned)}

特征范数:
  Video: {np.linalg.norm(v_feat_aligned):.2f}
  Audio: {np.linalg.norm(a_feat_aligned):.2f}
  Fusion: {np.linalg.norm(f_feat_aligned):.2f}

相似度:
  Video-Audio: {v_a_sim:.3f}
  Video-Fusion: {v_f_sim:.3f}
  Audio-Fusion: {a_f_sim:.3f}

估计权重:
  Video: {v_weight:.1%}
  Audio: {a_weight:.1%}
            """
            axes[1, 2].text(0.1, 0.5, fusion_text, fontsize=10,
                            verticalalignment='center', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            axes[0, 0].axis('off')
            axes[0, 1].axis('off')
            axes[0, 2].axis('off')
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
            axes[1, 2].text(0.1, 0.5, "仅有单一模态可用，跳过融合分析", fontsize=12)

        plt.suptitle(f'多模态融合分析: {name}', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fusion' / f'{name}_fusion.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 融合分析已保存")

    def _visualize_prediction(
            self,
            probs: np.ndarray,
            pred: int,
            label: Optional[int],
            name: str
    ):
        """预测结果可视化"""
        print("\n📊 6. 预测分析...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top-5预测
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_probs = probs[top5_idx]
        top5_names = [self.class_names[i] for i in top5_idx]

        colors = ['green' if i == pred else 'gray' for i in top5_idx]
        axes[0, 0].barh(range(5), top5_probs, color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(5))
        axes[0, 0].set_yticklabels(top5_names)
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_title('Top-5 Predictions', fontweight='bold')
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 所有类别概率
        axes[0, 1].bar(range(self.num_classes), probs, alpha=0.7)
        axes[0, 1].bar(pred, probs[pred], color='green', alpha=0.9, label='Predicted')
        if label is not None:
            axes[0, 1].bar(label, probs[label], color='blue', alpha=0.5, label='True')
        axes[0, 1].set_xticks(range(self.num_classes))
        axes[0, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_title('All Classes', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 预测置信度分析
        pred_entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(self.num_classes)
        confidence = probs[pred]

        metrics = {
            'Confidence': confidence,
            'Entropy': pred_entropy / max_entropy,  # 归一化
            'Top-2 Gap': top5_probs[0] - top5_probs[1] if len(top5_probs) > 1 else 0.0,
        }

        axes[1, 0].bar(metrics.keys(), metrics.values(), alpha=0.7)
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Prediction Metrics', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

        # 预测摘要
        axes[1, 1].axis('off')
        summary_text = f"""
预测摘要:

预测类别: {self.class_names[pred]}
置信度: {confidence:.3f}

"""
        if label is not None:
            summary_text += f"""真实标签: {self.class_names[label]}
结果: {'✓ 正确' if pred == label else '✗ 错误'}

"""

        summary_text += f"""Top-5:
  1. {top5_names[0]}: {top5_probs[0]:.3f}
  2. {top5_names[1]}: {top5_probs[1]:.3f}
  3. {top5_names[2]}: {top5_probs[2]:.3f}
  4. {top5_names[3]}: {top5_probs[3]:.3f}
  5. {top5_names[4]}: {top5_probs[4]:.3f}

不确定性:
  熵: {pred_entropy:.3f} / {max_entropy:.3f}
  归一化熵: {pred_entropy / max_entropy:.3f}
  Top-2差距: {metrics['Top-2 Gap']:.3f}

评估: {'✓ 高置信' if confidence > 0.8 else '⚠ 中等置信' if confidence > 0.5 else '✗ 低置信'}
        """

        color = 'lightgreen' if confidence > 0.8 else 'lightyellow' if confidence > 0.5 else 'lightcoral'
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10,
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

        plt.suptitle(f'预测分析: {name}', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name}_prediction.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 预测分析已保存")

    def _create_summary_figure(
            self,
            video: torch.Tensor,
            audio: torch.Tensor,
            outputs: Dict,
            probs: np.ndarray,
            pred: int,
            label: Optional[int],
            name: str
    ):
        """创建总结图"""
        print("\n📋 7. 生成总结图...")

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 第一行：输入数据
        video_np = video.cpu().numpy()
        audio_np = audio.cpu().numpy()

        # 选择中间帧
        mid_frame = video_np[len(video_np) // 2]
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(mid_frame.transpose(1, 2, 0))
        ax1.set_title('Input Video Frame', fontweight='bold')
        ax1.axis('off')

        # 音频光谱
        ax2 = fig.add_subplot(gs[0, 1])
        audio_mean = audio_np.mean(axis=0)
        im = ax2.imshow(audio_mean, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('Audio Spectrogram', fontweight='bold')
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Mel bins')
        plt.colorbar(im, ax=ax2, fraction=0.046)

        # CAVA门控
        ax3 = fig.add_subplot(gs[0, 2:])
        if isinstance(outputs, dict) and 'causal_gate' in outputs:
            gate = outputs['causal_gate'][0].cpu().numpy()
            if gate.ndim > 1:
                gate = gate.mean(axis=1)
            ax3.plot(range(len(gate)), gate, 'o-', linewidth=2)
            ax3.fill_between(range(len(gate)), 0, gate, alpha=0.3)
            ax3.set_title('CAVA Causal Gate', fontweight='bold')
            ax3.set_xlabel('Time step')
            ax3.set_ylabel('Gate value')
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'CAVA Gate\nNot Available',
                     ha='center', va='center', fontsize=12)
            ax3.axis('off')

        # 第二行：特征和相似度
        if isinstance(outputs, dict):
            v_feat = outputs.get('video_proj', outputs.get('video_emb'))
            a_feat = outputs.get('audio_aligned', outputs.get('audio_emb'))

            if v_feat is not None:
                ax4 = fig.add_subplot(gs[1, 0])
                v_np = v_feat[0].cpu().numpy()
                im = ax4.imshow(v_np.T, aspect='auto', cmap='Reds')
                ax4.set_title('Video Features', fontweight='bold')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Dim')
                plt.colorbar(im, ax=ax4, fraction=0.046)

            if a_feat is not None:
                ax5 = fig.add_subplot(gs[1, 1])
                a_np = a_feat[0].cpu().numpy()
                im = ax5.imshow(a_np.T, aspect='auto', cmap='Blues')
                ax5.set_title('Audio Features', fontweight='bold')
                ax5.set_xlabel('Time')
                ax5.set_ylabel('Dim')
                plt.colorbar(im, ax=ax5, fraction=0.046)

            # 相似度
            if v_feat is not None and a_feat is not None:
                ax6 = fig.add_subplot(gs[1, 2:])
                v_np = v_feat[0].cpu().numpy()
                a_np = a_feat[0].cpu().numpy()
                T = min(len(v_np), len(a_np))
                v_norm = v_np[:T] / (np.linalg.norm(v_np[:T], axis=1, keepdims=True) + 1e-8)
                a_norm = a_np[:T] / (np.linalg.norm(a_np[:T], axis=1, keepdims=True) + 1e-8)
                sim = np.sum(v_norm * a_norm, axis=1)
                ax6.plot(range(T), sim, 'o-', linewidth=2)
                ax6.fill_between(range(T), 0, sim, alpha=0.3)
                ax6.set_title('Modality Similarity', fontweight='bold')
                ax6.set_xlabel('Time step')
                ax6.set_ylabel('Cosine Similarity')
                ax6.grid(True, alpha=0.3)

        # 第三行：预测结果
        ax7 = fig.add_subplot(gs[2, :2])
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_probs = probs[top5_idx]
        top5_names = [self.class_names[i] for i in top5_idx]
        colors = ['green' if i == pred else 'gray' for i in top5_idx]
        ax7.barh(range(5), top5_probs, color=colors, alpha=0.7)
        ax7.set_yticks(range(5))
        ax7.set_yticklabels(top5_names)
        ax7.set_xlabel('Probability')
        ax7.set_title('Top-5 Predictions', fontweight='bold')
        ax7.set_xlim([0, 1])
        ax7.grid(axis='x', alpha=0.3)

        # 摘要信息
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')

        confidence = probs[pred]
        summary = f"""
推理摘要

预测: {self.class_names[pred]}
置信度: {confidence:.1%}

"""
        if label is not None:
            summary += f"""真实: {self.class_names[label]}
{'✓ 正确' if pred == label else '✗ 错误'}

"""

        summary += f"""Top-3:
1. {top5_names[0]}: {top5_probs[0]:.1%}
2. {top5_names[1]}: {top5_probs[1]:.1%}
3. {top5_names[2]}: {top5_probs[2]:.1%}

模型状态: 正常
        """

        color = 'lightgreen' if confidence > 0.8 else 'lightyellow'
        ax8.text(0.1, 0.5, summary, fontsize=11,
                 verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

        plt.suptitle(f'推理总结: {name}', fontsize=16, fontweight='bold')
        plt.savefig(self.output_dir / f'{name}_summary.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 总结图已保存")


def main():
    parser = argparse.ArgumentParser(description='单样本推理可视化')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='./inference_vis',
                        help='输出目录')

    # 输入方式1：从数据集
    parser.add_argument('--dataset', type=str,
                        help='数据集CSV文件（用于选择样本）')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='样本索引')

    # 输入方式2：直接指定文件
    parser.add_argument('--video', type=str,
                        help='视频文件路径')
    parser.add_argument('--audio', type=str,
                        help='音频文件路径')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🎬 单样本推理可视化工具")
    print("=" * 60)

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")

    # 加载模型
    print(f"📦 加载模型: {args.checkpoint}")
    model_cfg = cfg.get("model", {})
    model_cfg["num_classes"] = cfg["data"]["num_classes"]

    model = EnhancedAVTopDetector({
        "model": model_cfg,
        "fusion": model_cfg.get("fusion", {}),
        "cava": cfg.get("cava", {})
    }).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
    model.eval()
    print(f"✅ 模型加载成功")

    # 加载数据
    if args.dataset:
        print(f"📊 从数据集加载样本 {args.sample_idx}...")
        dataset = AVFromCSV(
            args.dataset,
            cfg["data"].get("data_root"),
            cfg["data"]["num_classes"],
            cfg["data"]["class_names"],
            cfg.get("video", {}),
            cfg.get("audio", {}),
            is_unlabeled=False
        )

        video, audio, label = dataset[args.sample_idx][:3]
        sample_name = f"sample_{args.sample_idx}"
        label = label.item() if torch.is_tensor(label) else label
    elif args.video and args.audio:
        print(f"📂 从文件加载...")
        print(f"   视频: {args.video}")
        print(f"   音频: {args.audio}")
        # TODO: 如需支持文件直读，可在此实现
        print("❌ 暂不支持从文件直接加载，请使用 --dataset 参数")
        return
    else:
        print("❌ 请指定 --dataset 或 --video/--audio")
        return

    # 创建可视化器
    visualizer = InferenceVisualizer(
        model=model,
        class_names=cfg["data"]["class_names"],
        device=device,
        output_dir=args.output
    )

    # 执行可视化
    visualizer.visualize_sample(video, audio, label, sample_name)

    print("\n" + "=" * 60)
    print("🎉 可视化完成！")
    print(f"📁 结果保存在: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
