#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证和可视化脚本 - 全面展示模型性能和核心创新

功能：
1. 验证集评估 - 混淆矩阵、ROC曲线、性能指标
2. CAVA模块可视化 - 音视频对齐、延迟估计、门控机制
3. MLPR模块可视化 - 权重分布、置信度分析、历史统计
4. 特征空间可视化 - t-SNE、注意力图、融合过程
5. 时序分析 - 逐帧特征演化

使用方法：
    python validate_and_visualize.py \
        --checkpoint runs/fixed_exp/checkpoints/best_f1.pth \
        --config selfsup_sota.yaml \
        --output ./visualizations \
        --num_samples 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# 可视化库
import matplotlib

matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 科学计算和评估
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    accuracy_score, f1_score, precision_recall_curve
)
from sklearn.manifold import TSNE
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter

# 导入模型和数据集
from enhanced_detector import EnhancedAVTopDetector
from dataset import AVFromCSV, safe_collate_fn

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)


# 配置Windows兼容的中文字体
def setup_chinese_font():
    """配置中文字体，支持Windows/macOS/Linux"""
    import platform
    system = platform.system()

    if system == 'Windows':
        font_options = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':
        font_options = ['PingFang SC', 'Heiti SC', 'STHeiti']
    else:
        font_options = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

    try:
        import matplotlib.font_manager as fm
        available_fonts = set([f.name for f in fm.fontManager.ttflist])

        for font in font_options:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用字体: {font}")
                return font
    except Exception as e:
        print(f"⚠️  字体配置失败: {e}")

    plt.rcParams['axes.unicode_minus'] = False
    return None


setup_chinese_font()


class ModelVisualizer:
    """模型可视化器 - 全面展示模型行为和性能"""

    def __init__(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            class_names: List[str],
            device: torch.device,
            output_dir: str
    ):
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'cava').mkdir(exist_ok=True)
        (self.output_dir / 'mlpr').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)

        # 收集的数据
        self.predictions = []
        self.ground_truths = []
        self.probabilities = []
        self.features_data = {
            'video_features': [],
            'audio_features': [],
            'fusion_features': [],
            'cava_gates': [],
            'cava_delays': [],
            'attention_maps': [],
        }

        print(f"📁 输出目录: {self.output_dir}")

    @torch.no_grad()
    def collect_predictions(self, num_samples: Optional[int] = None):
        """收集模型预测和中间特征"""
        print("\n" + "=" * 60)
        print("📊 第一步：收集预测和特征")
        print("=" * 60)

        self.model.eval()
        sample_count = 0

        pbar = tqdm(self.dataloader, desc="收集数据")
        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                video, audio, labels = batch[:3]
            else:
                continue

            video = video.to(self.device)
            audio = audio.to(self.device)
            labels = labels.argmax(dim=1) if labels.ndim == 2 else labels

            # 前向传播（获取完整输出）
            outputs = self.model(video, audio, return_aux=True)

            # 提取预测
            if isinstance(outputs, dict):
                logits = outputs.get('clip_logits', list(outputs.values())[0])
            else:
                logits = outputs

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            self.predictions.extend(preds)
            self.ground_truths.extend(labels.cpu().numpy())
            self.probabilities.extend(probs)

            # 收集中间特征
            if isinstance(outputs, dict):
                # 视频特征
                if 'video_proj' in outputs or 'video_emb' in outputs:
                    v_feat = outputs.get('video_proj', outputs.get('video_emb'))
                    if v_feat is not None:
                        self.features_data['video_features'].append(
                            v_feat.mean(dim=1).cpu().numpy()  # [B, D]
                        )

                # 音频特征
                if 'audio_aligned' in outputs or 'audio_emb' in outputs:
                    a_feat = outputs.get('audio_aligned', outputs.get('audio_emb'))
                    if a_feat is not None:
                        self.features_data['audio_features'].append(
                            a_feat.mean(dim=1).cpu().numpy()
                        )

                # 融合特征
                if 'fusion_token' in outputs or 'fusion_out' in outputs:
                    f_feat = outputs.get('fusion_token', outputs.get('fusion_out'))
                    if f_feat is not None:
                        if f_feat.dim() > 2:
                            f_feat = f_feat.mean(dim=1)
                        self.features_data['fusion_features'].append(f_feat.cpu().numpy())

                # CAVA门控
                if 'causal_gate' in outputs and outputs['causal_gate'] is not None:
                    gate = outputs['causal_gate']
                    if gate.dim() > 2:
                        gate = gate.mean(dim=1)  # [B, T] or [B]
                    self.features_data['cava_gates'].append(gate.cpu().numpy())

                # CAVA延迟
                if 'delay_frames' in outputs and outputs['delay_frames'] is not None:
                    delay = outputs['delay_frames']
                    self.features_data['cava_delays'].append(delay.cpu().numpy())

            sample_count += len(labels)
            pbar.set_postfix({'samples': sample_count})

            if num_samples and sample_count >= num_samples:
                break

        # 转换为numpy数组
        self.predictions = np.array(self.predictions)
        self.ground_truths = np.array(self.ground_truths)
        self.probabilities = np.array(self.probabilities)

        # 合并特征
        for key in ['video_features', 'audio_features', 'fusion_features',
                    'cava_gates', 'cava_delays']:
            if self.features_data[key]:
                self.features_data[key] = np.concatenate(self.features_data[key], axis=0)
            else:
                self.features_data[key] = None

        print(f"✅ 收集完成: {len(self.predictions)} 个样本")
        print(f"   - 预测形状: {self.predictions.shape}")
        print(f"   - 概率形状: {self.probabilities.shape}")
        if self.features_data['video_features'] is not None:
            print(f"   - 视频特征: {self.features_data['video_features'].shape}")
        if self.features_data['audio_features'] is not None:
            print(f"   - 音频特征: {self.features_data['audio_features'].shape}")
        if self.features_data['cava_gates'] is not None:
            print(f"   - CAVA门控: {self.features_data['cava_gates'].shape}")
        if self.features_data['cava_delays'] is not None:
            print(f"   - CAVA延迟: {self.features_data['cava_delays'].shape}")

    def visualize_basic_metrics(self):
        """1. 基础性能指标可视化"""
        print("\n" + "=" * 60)
        print("📈 第二步：基础性能指标")
        print("=" * 60)

        # 计算指标
        acc = accuracy_score(self.ground_truths, self.predictions)
        f1_macro = f1_score(self.ground_truths, self.predictions, average='macro')
        f1_weighted = f1_score(self.ground_truths, self.predictions, average='weighted')

        # 每类指标
        report = classification_report(
            self.ground_truths, self.predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        # 保存报告
        report_path = self.output_dir / 'metrics' / 'classification_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✅ 整体准确率: {acc:.4f}")
        print(f"✅ 宏平均F1: {f1_macro:.4f}")
        print(f"✅ 加权F1: {f1_weighted:.4f}")

        # 混淆矩阵
        self._plot_confusion_matrix()

        # ROC曲线
        self._plot_roc_curves()

        # 每类性能条形图
        self._plot_per_class_metrics(report)

        print(f"💾 指标已保存到: {self.output_dir / 'metrics'}")

    def _plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.ground_truths, self.predictions)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # 原始计数
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('混淆矩阵 (计数)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('真实标签', fontsize=12)
        axes[0].set_xlabel('预测标签', fontsize=12)

        # 归一化百分比
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[1], cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
        axes[1].set_title('混淆矩阵 (归一化)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('真实标签', fontsize=12)
        axes[1].set_xlabel('预测标签', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics' / 'confusion_matrix.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 混淆矩阵已生成")

    def _plot_roc_curves(self):
        """绘制ROC曲线"""
        # 转换为one-hot
        y_true_oh = np.eye(self.num_classes)[self.ground_truths]

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        aucs = []
        for i, class_name in enumerate(self.class_names):
            if i >= len(axes):
                break

            fpr, tpr, _ = roc_curve(y_true_oh[:, i], self.probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            axes[i].plot(fpr, tpr, color='darkorange', lw=2,
                         label=f'AUC = {roc_auc:.3f}')
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('假阳性率')
            axes[i].set_ylabel('真阳性率')
            axes[i].set_title(f'{class_name}', fontweight='bold')
            axes[i].legend(loc="lower right")
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(f'ROC曲线 (平均AUC = {np.mean(aucs):.3f})',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics' / 'roc_curves.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ ROC曲线已生成")

    def _plot_per_class_metrics(self, report: Dict):
        """每类性能条形图"""
        classes = [c for c in self.class_names if c in report]
        precisions = [report[c]['precision'] for c in classes]
        recalls = [report[c]['recall'] for c in classes]
        f1s = [report[c]['f1-score'] for c in classes]
        supports = [report[c]['support'] for c in classes]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        x = np.arange(len(classes))
        width = 0.25

        # Precision, Recall, F1对比
        axes[0, 0].bar(x - width, precisions, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recalls, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
        axes[0, 0].set_ylabel('分数')
        axes[0, 0].set_title('每类性能指标对比', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim([0, 1])

        # F1-Score排序
        sorted_idx = np.argsort(f1s)
        axes[0, 1].barh(range(len(classes)), [f1s[i] for i in sorted_idx],
                        color=plt.cm.RdYlGn([f1s[i] for i in sorted_idx]))
        axes[0, 1].set_yticks(range(len(classes)))
        axes[0, 1].set_yticklabels([classes[i] for i in sorted_idx])
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_title('F1-Score排序', fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        axes[0, 1].set_xlim([0, 1])

        # 样本数量
        axes[1, 0].bar(x, supports, alpha=0.7, color='steelblue')
        axes[1, 0].set_ylabel('样本数量')
        axes[1, 0].set_title('各类样本分布', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 加权F1 vs 样本数
        axes[1, 1].scatter(supports, f1s, s=100, alpha=0.6, c=f1s,
                           cmap='RdYlGn', vmin=0, vmax=1)
        for i, cls in enumerate(classes):
            axes[1, 1].annotate(cls, (supports[i], f1s[i]),
                                fontsize=8, alpha=0.7)
        axes[1, 1].set_xlabel('样本数量')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('F1与样本数关系', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics' / 'per_class_metrics.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 每类性能图已生成")

    def visualize_cava_module(self):
        """2. CAVA模块可视化 - 音视频因果对齐"""
        print("\n" + "=" * 60)
        print("🎯 第三步：CAVA模块可视化")
        print("=" * 60)

        if self.features_data['cava_gates'] is None:
            print("⚠️  CAVA门控数据不可用，跳过")
            return

        # 1. 门控分布分析
        self._plot_cava_gate_distribution()

        # 2. 延迟估计分析
        if self.features_data['cava_delays'] is not None:
            self._plot_cava_delay_distribution()

        # 3. 门控与预测置信度关系
        self._plot_gate_confidence_relation()

        # 4. 不同类别的对齐模式
        self._plot_alignment_patterns_per_class()

        print(f"💾 CAVA可视化已保存到: {self.output_dir / 'cava'}")

    def _plot_cava_gate_distribution(self):
        """CAVA门控分布"""
        gates = self.features_data['cava_gates']

        if gates.ndim == 1:
            gates = gates.reshape(-1, 1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 整体分布
        axes[0, 0].hist(gates.flatten(), bins=50, alpha=0.7,
                        color='steelblue', edgecolor='black')
        axes[0, 0].axvline(gates.mean(), color='red', linestyle='--',
                           linewidth=2, label=f'Mean={gates.mean():.3f}')
        axes[0, 0].axvline(np.median(gates), color='green', linestyle='--',
                           linewidth=2, label=f'Median={np.median(gates):.3f}')
        axes[0, 0].set_xlabel('门控值')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('CAVA门控分布', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 按类别分布
        unique_classes = np.unique(self.ground_truths)
        for cls in unique_classes[:6]:  # 前6类
            mask = self.ground_truths == cls
            cls_gates = gates[mask].flatten()
            axes[0, 1].hist(cls_gates, bins=30, alpha=0.5,
                            label=self.class_names[cls])
        axes[0, 1].set_xlabel('门控值')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('各类别门控分布', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 门控vs预测正确性
        correct = (self.predictions == self.ground_truths)
        gates_correct = gates[correct].flatten()
        gates_wrong = gates[~correct].flatten()

        axes[1, 0].hist([gates_correct, gates_wrong], bins=30,
                        label=['正确预测', '错误预测'],
                        alpha=0.7, color=['green', 'red'])
        axes[1, 0].set_xlabel('门控值')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('门控分布：正确vs错误预测', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 箱线图对比
        data_to_plot = [gates_correct, gates_wrong]
        bp = axes[1, 1].boxplot(data_to_plot, labels=['正确', '错误'],
                                patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        axes[1, 1].set_ylabel('门控值')
        axes[1, 1].set_title('门控分布箱线图', fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cava' / 'gate_distribution.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ CAVA门控分布已生成")

    def _plot_cava_delay_distribution(self):
        """CAVA延迟估计分布"""
        delays = self.features_data['cava_delays']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 延迟分布
        axes[0, 0].hist(delays, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 0].axvline(delays.mean(), color='red', linestyle='--',
                           linewidth=2, label=f'Mean={delays.mean():.2f}')
        axes[0, 0].set_xlabel('延迟 (帧)')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('CAVA延迟估计分布', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 延迟vs门控
        gates = self.features_data['cava_gates']
        if gates.ndim > 1:
            gates = gates.mean(axis=1)

        axes[0, 1].scatter(delays, gates, alpha=0.5, s=20)
        axes[0, 1].set_xlabel('延迟 (帧)')
        axes[0, 1].set_ylabel('平均门控值')
        axes[0, 1].set_title('延迟 vs 门控关系', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 各类别延迟
        unique_classes = np.unique(self.ground_truths)
        class_delays = [delays[self.ground_truths == cls] for cls in unique_classes]
        bp = axes[1, 0].boxplot(class_delays,
                                labels=[self.class_names[i] for i in unique_classes],
                                patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1, 0].set_xticklabels([self.class_names[i] for i in unique_classes],
                                   rotation=45, ha='right')
        axes[1, 0].set_ylabel('延迟 (帧)')
        axes[1, 0].set_title('各类别延迟分布', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 延迟热图
        delay_matrix = np.zeros((self.num_classes, 50))
        for cls in range(self.num_classes):
            mask = self.ground_truths == cls
            if mask.sum() > 0:
                hist, _ = np.histogram(delays[mask], bins=50, range=(delays.min(), delays.max()))
                delay_matrix[cls] = hist

        im = axes[1, 1].imshow(delay_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        axes[1, 1].set_xlabel('延迟区间')
        axes[1, 1].set_ylabel('类别')
        axes[1, 1].set_yticks(range(self.num_classes))
        axes[1, 1].set_yticklabels(self.class_names)
        axes[1, 1].set_title('类别-延迟热图', fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1], label='样本数')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cava' / 'delay_distribution.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ CAVA延迟分布已生成")

    def _plot_gate_confidence_relation(self):
        """门控与预测置信度关系"""
        gates = self.features_data['cava_gates']
        if gates.ndim > 1:
            gates = gates.mean(axis=1)

        # 预测置信度
        max_probs = self.probabilities.max(axis=1)

        # 预测熵
        pred_entropy = entropy(self.probabilities.T)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 门控 vs 置信度
        axes[0, 0].scatter(gates, max_probs, alpha=0.5, s=20, c=self.predictions,
                           cmap='tab20')
        axes[0, 0].set_xlabel('门控值')
        axes[0, 0].set_ylabel('预测置信度')
        axes[0, 0].set_title('门控 vs 预测置信度', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # 门控 vs 熵
        axes[0, 1].scatter(gates, pred_entropy, alpha=0.5, s=20,
                           c=self.predictions, cmap='tab20')
        axes[0, 1].set_xlabel('门控值')
        axes[0, 1].set_ylabel('预测熵')
        axes[0, 1].set_title('门控 vs 预测不确定性', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 分组分析：高门控vs低门控
        gate_threshold = np.median(gates)
        high_gate_mask = gates > gate_threshold
        low_gate_mask = gates <= gate_threshold

        high_gate_conf = max_probs[high_gate_mask]
        low_gate_conf = max_probs[low_gate_mask]

        axes[1, 0].hist([high_gate_conf, low_gate_conf], bins=30,
                        label=[f'高门控(>{gate_threshold:.2f})',
                               f'低门控(<={gate_threshold:.2f})'],
                        alpha=0.7, color=['green', 'orange'])
        axes[1, 0].set_xlabel('预测置信度')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('置信度分布：高门控vs低门控', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 2D密度图
        from scipy.stats import gaussian_kde
        if len(gates) > 100:
            xy = np.vstack([gates, max_probs])
            z = gaussian_kde(xy)(xy)
            axes[1, 1].scatter(gates, max_probs, c=z, s=20, cmap='viridis', alpha=0.5)
            axes[1, 1].set_xlabel('门控值')
            axes[1, 1].set_ylabel('预测置信度')
            axes[1, 1].set_title('门控-置信度密度图', fontweight='bold')
            plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='密度')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cava' / 'gate_confidence_relation.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 门控-置信度关系图已生成")

    def _plot_alignment_patterns_per_class(self):
        """各类别的对齐模式"""
        gates = self.features_data['cava_gates']
        if gates.ndim == 1:
            gates = gates.reshape(-1, 1)

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        for i, cls in enumerate(range(self.num_classes)):
            if i >= len(axes):
                break

            mask = self.ground_truths == cls
            cls_gates = gates[mask]

            if cls_gates.shape[1] > 1:
                # 多时间步：显示热图
                im = axes[i].imshow(cls_gates[:min(50, len(cls_gates))].T,
                                    aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
                axes[i].set_xlabel('样本')
                axes[i].set_ylabel('时间步')
                plt.colorbar(im, ax=axes[i], fraction=0.046)
            else:
                # 单值：显示分布
                axes[i].hist(cls_gates.flatten(), bins=20, alpha=0.7,
                             color='steelblue', edgecolor='black')
                axes[i].set_xlabel('门控值')
                axes[i].set_ylabel('频数')

            axes[i].set_title(f'{self.class_names[cls]}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)

        plt.suptitle('各类别CAVA对齐模式', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cava' / 'alignment_patterns.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 各类别对齐模式已生成")

    def visualize_feature_space(self):
        """3. 特征空间可视化"""
        print("\n" + "=" * 60)
        print("🗺️  第四步：特征空间可视化")
        print("=" * 60)

        # t-SNE降维
        self._plot_tsne_visualization()

        # 模态融合分析
        if (self.features_data['video_features'] is not None and
                self.features_data['audio_features'] is not None):
            self._plot_modality_fusion()

        # 特征相似度矩阵
        self._plot_feature_similarity()

        print(f"💾 特征空间可视化已保存到: {self.output_dir / 'features'}")

    def _plot_tsne_visualization(self):
        """t-SNE特征空间可视化"""
        # 尝试使用融合特征，回退到其他特征
        features = None
        feature_name = ""

        if self.features_data['fusion_features'] is not None:
            features = self.features_data['fusion_features']
            feature_name = "Fusion"
        elif self.features_data['video_features'] is not None:
            features = self.features_data['video_features']
            feature_name = "Video"
        elif self.features_data['audio_features'] is not None:
            features = self.features_data['audio_features']
            feature_name = "Audio"
        else:
            print("  ⚠️  没有可用的特征进行t-SNE可视化")
            return

        print(f"  使用 {feature_name} 特征进行t-SNE...")

        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
        features_2d = tsne.fit_transform(features)

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # 按真实标签着色
        scatter1 = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1],
                                      c=self.ground_truths, cmap='tab20',
                                      s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 0].set_title(f't-SNE: {feature_name}特征 (真实标签)', fontweight='bold')
        axes[0, 0].set_xlabel('t-SNE 维度 1')
        axes[0, 0].set_ylabel('t-SNE 维度 2')
        legend1 = axes[0, 0].legend(*scatter1.legend_elements(),
                                    title="类别", loc="best", ncol=2)
        axes[0, 0].add_artist(legend1)
        axes[0, 0].grid(True, alpha=0.3)

        # 按预测标签着色
        scatter2 = axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1],
                                      c=self.predictions, cmap='tab20',
                                      s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 1].set_title(f't-SNE: {feature_name}特征 (预测标签)', fontweight='bold')
        axes[0, 1].set_xlabel('t-SNE 维度 1')
        axes[0, 1].set_ylabel('t-SNE 维度 2')
        axes[0, 1].grid(True, alpha=0.3)

        # 按预测置信度着色
        max_probs = self.probabilities.max(axis=1)
        scatter3 = axes[1, 0].scatter(features_2d[:, 0], features_2d[:, 1],
                                      c=max_probs, cmap='RdYlGn',
                                      s=50, alpha=0.6, edgecolors='k', linewidth=0.5,
                                      vmin=0, vmax=1)
        axes[1, 0].set_title('t-SNE: 预测置信度', fontweight='bold')
        axes[1, 0].set_xlabel('t-SNE 维度 1')
        axes[1, 0].set_ylabel('t-SNE 维度 2')
        plt.colorbar(scatter3, ax=axes[1, 0], label='置信度')
        axes[1, 0].grid(True, alpha=0.3)

        # 标注错误预测
        correct = (self.predictions == self.ground_truths)
        axes[1, 1].scatter(features_2d[correct, 0], features_2d[correct, 1],
                           c='green', s=30, alpha=0.3, label='正确')
        axes[1, 1].scatter(features_2d[~correct, 0], features_2d[~correct, 1],
                           c='red', s=50, alpha=0.8, marker='x', label='错误')
        axes[1, 1].set_title('t-SNE: 预测正确性', fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE 维度 1')
        axes[1, 1].set_ylabel('t-SNE 维度 2')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'features' / 'tsne_visualization.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ t-SNE可视化已生成")

    def _plot_modality_fusion(self):
        """模态融合分析"""
        v_feat = self.features_data['video_features']
        a_feat = self.features_data['audio_features']

        # 计算模态间相似度
        v_norm = v_feat / (np.linalg.norm(v_feat, axis=1, keepdims=True) + 1e-8)
        a_norm = a_feat / (np.linalg.norm(a_feat, axis=1, keepdims=True) + 1e-8)
        similarity = np.sum(v_norm * a_norm, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 相似度分布
        axes[0, 0].hist(similarity, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].axvline(similarity.mean(), color='red', linestyle='--',
                           linewidth=2, label=f'Mean={similarity.mean():.3f}')
        axes[0, 0].set_xlabel('余弦相似度')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('音视频特征相似度分布', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 相似度vs预测置信度
        max_probs = self.probabilities.max(axis=1)
        axes[0, 1].scatter(similarity, max_probs, alpha=0.5, s=20,
                           c=self.predictions, cmap='tab20')
        axes[0, 1].set_xlabel('模态相似度')
        axes[0, 1].set_ylabel('预测置信度')
        axes[0, 1].set_title('模态相似度 vs 预测置信度', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 各类别相似度
        class_similarities = [similarity[self.ground_truths == cls]
                              for cls in range(self.num_classes)]
        bp = axes[1, 0].boxplot(class_similarities,
                                labels=self.class_names,
                                patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1, 0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('相似度')
        axes[1, 0].set_title('各类别模态相似度', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 正确vs错误预测的相似度
        correct = (self.predictions == self.ground_truths)
        axes[1, 1].hist([similarity[correct], similarity[~correct]],
                        bins=30, label=['正确', '错误'],
                        alpha=0.7, color=['green', 'red'])
        axes[1, 1].set_xlabel('模态相似度')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('模态相似度：正确vs错误预测', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'features' / 'modality_fusion.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 模态融合分析已生成")

    def _plot_feature_similarity(self):
        """特征相似度矩阵"""
        # 使用融合特征或视频特征
        features = (self.features_data['fusion_features']
                    if self.features_data['fusion_features'] is not None
                    else self.features_data['video_features'])

        if features is None:
            return

        # 计算类中心
        class_centers = []
        for cls in range(self.num_classes):
            mask = self.ground_truths == cls
            if mask.sum() > 0:
                class_centers.append(features[mask].mean(axis=0))
            else:
                class_centers.append(np.zeros(features.shape[1]))
        class_centers = np.array(class_centers)

        # 计算类间相似度
        class_centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(class_centers_norm, class_centers_norm.T)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # 相似度矩阵
        im1 = axes[0].imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_xticks(range(self.num_classes))
        axes[0].set_yticks(range(self.num_classes))
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].set_yticklabels(self.class_names)
        axes[0].set_title('类间特征相似度矩阵', fontweight='bold')

        # 添加数值标注
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                text = axes[0].text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                    ha="center", va="center",
                                    color="white" if abs(similarity_matrix[i, j]) > 0.5 else "black",
                                    fontsize=8)

        plt.colorbar(im1, ax=axes[0], label='余弦相似度')

        # 距离矩阵（1 - 相似度）
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)

        im2 = axes[1].imshow(distance_matrix, cmap='YlOrRd', vmin=0, vmax=2)
        axes[1].set_xticks(range(self.num_classes))
        axes[1].set_yticks(range(self.num_classes))
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].set_yticklabels(self.class_names)
        axes[1].set_title('类间特征距离矩阵', fontweight='bold')
        plt.colorbar(im2, ax=axes[1], label='距离 (1-相似度)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'features' / 'class_similarity_matrix.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("  ✓ 特征相似度矩阵已生成")

    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("📄 第五步：生成总结报告")
        print("=" * 60)

        acc = accuracy_score(self.ground_truths, self.predictions)
        f1_macro = f1_score(self.ground_truths, self.predictions, average='macro')

        report = {
            "overall_metrics": {
                "accuracy": float(acc),
                "f1_macro": float(f1_macro),
                "num_samples": len(self.predictions),
                "num_classes": self.num_classes,
            },
            "cava_statistics": {},
            "feature_statistics": {},
        }

        # CAVA统计
        if self.features_data['cava_gates'] is not None:
            gates = self.features_data['cava_gates']
            report["cava_statistics"]["gate_mean"] = float(gates.mean())
            report["cava_statistics"]["gate_std"] = float(gates.std())
            report["cava_statistics"]["gate_min"] = float(gates.min())
            report["cava_statistics"]["gate_max"] = float(gates.max())

        if self.features_data['cava_delays'] is not None:
            delays = self.features_data['cava_delays']
            report["cava_statistics"]["delay_mean"] = float(delays.mean())
            report["cava_statistics"]["delay_std"] = float(delays.std())

        # 保存报告
        report_path = self.output_dir / 'summary_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✅ 总结报告已保存: {report_path}")

        # 创建README
        self._create_readme()

    def _create_readme(self):
        """创建README文件"""
        readme_content = f"""# 模型验证和可视化结果

## 整体性能

- **准确率**: {accuracy_score(self.ground_truths, self.predictions):.4f}
- **宏平均F1**: {f1_score(self.ground_truths, self.predictions, average='macro'):.4f}
- **样本数量**: {len(self.predictions)}

## 文件结构

```
{self.output_dir.name}/
├── metrics/                    # 性能指标
│   ├── confusion_matrix.png    # 混淆矩阵
│   ├── roc_curves.png          # ROC曲线
│   ├── per_class_metrics.png   # 每类性能
│   └── classification_report.json
│
├── cava/                       # CAVA模块可视化
│   ├── gate_distribution.png   # 门控分布
│   ├── delay_distribution.png  # 延迟分布
│   ├── gate_confidence_relation.png
│   └── alignment_patterns.png  # 对齐模式
│
├── features/                   # 特征空间
│   ├── tsne_visualization.png  # t-SNE降维
│   ├── modality_fusion.png     # 模态融合
│   └── class_similarity_matrix.png
│
└── summary_report.json         # 总结报告
```

## 核心创新可视化

### 1. CAVA (音视频因果对齐)
- 门控机制分布和作用
- 延迟估计的准确性
- 对齐质量与预测性能的关系

### 2. 特征空间分析
- 多模态特征融合效果
- 类间可分性
- t-SNE降维可视化

## 使用方法

1. 查看 `metrics/` 了解整体性能
2. 查看 `cava/` 了解音视频对齐效果
3. 查看 `features/` 了解特征学习质量

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✅ README已创建: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description='模型验证和可视化')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='./visualizations',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='使用的样本数量（None=全部）')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🎨 模型验证和可视化工具")
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
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    print(f"✅ 模型加载成功 (Epoch {checkpoint.get('epoch', '?')})")

    # 加载数据
    print("📊 加载验证数据...")
    data_cfg = cfg["data"]
    val_dataset = AVFromCSV(
        data_cfg["val_csv"],
        data_cfg.get("data_root"),
        data_cfg["num_classes"],
        data_cfg["class_names"],
        cfg.get("video", {}),
        cfg.get("audio", {}),
        is_unlabeled=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate_fn
    )
    print(f"✅ 数据加载完成: {len(val_dataset)} 个样本")

    # 创建可视化器
    visualizer = ModelVisualizer(
        model=model,
        dataloader=val_loader,
        class_names=data_cfg["class_names"],
        device=device,
        output_dir=args.output
    )

    # 执行可视化流程
    visualizer.collect_predictions(num_samples=args.num_samples)
    visualizer.visualize_basic_metrics()
    visualizer.visualize_cava_module()
    visualizer.visualize_feature_space()
    visualizer.generate_summary_report()

    print("\n" + "=" * 60)
    print("🎉 可视化完成！")
    print(f"📁 结果保存在: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 添加缺失的导入
    try:
        import pandas as pd
    except ImportError:
        import datetime


        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    class FakeTimestamp:
                        def strftime(self, fmt):
                            return datetime.datetime.now().strftime(fmt)

                    return FakeTimestamp()

    main()