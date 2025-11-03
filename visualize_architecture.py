#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型架构和特征维度可视化

功能：
1. 展示模型架构流程图
2. 特征维度变化详表
3. CAVA和融合模块详图
4. 参数量统计饼图

使用方法：
    python visualize_architecture.py \
        --checkpoint runs/fixed_exp/checkpoints/best_f1.pth \
        --config selfsup_sota.yaml \
        --output ./model_architecture.png
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# 配置中文字体
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

from enhanced_detector import EnhancedAVTopDetector


class ArchitectureVisualizer:
    """模型架构可视化器"""

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.model_cfg = config.get('model', {})
        self.video_cfg = config.get('video', {})
        self.audio_cfg = config.get('audio', {})
        self.cava_cfg = config.get('cava', {})

        # 提取关键参数
        self.num_classes = self.model_cfg.get('num_classes', 11)
        self.video_frames = self.video_cfg.get('num_frames', 8)
        self.audio_frames = self.audio_cfg.get('num_frames', 8)
        self.hidden_dim = self.model_cfg.get('hidden_dim', 256)
        self.fusion_dim = self.model_cfg.get('fusion_dim', 512)

    def count_parameters(self, module: nn.Module = None) -> int:
        """统计模块参数量"""
        if module is None:
            module = self.model
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def extract_feature_dims(self) -> Dict:
        """提取特征维度信息"""
        dims = {}

        # 输入维度
        dims['video_input'] = f"[B,{self.video_frames},3,224,224]"
        dims['audio_input'] = f"[B,{self.audio_frames},80,201]"

        # 编码器输出
        dims['video_backbone'] = f"[B,{self.video_frames},768]"
        dims['audio_backbone'] = f"[B,{self.audio_frames},768]"

        # 投影后
        dims['video_projection'] = f"[B,{self.video_frames},{self.hidden_dim}]"
        dims['audio_projection'] = f"[B,{self.audio_frames},{self.hidden_dim}]"

        # CAVA
        dims['cava_delay'] = "[B,1]"
        dims['cava_gate'] = f"[B,{self.audio_frames}]"
        dims['audio_aligned'] = f"[B,{self.audio_frames},{self.hidden_dim}]"

        # 融合
        dims['fusion_token'] = f"[B,1,{self.fusion_dim}]"

        # 输出
        dims['logits'] = f"[B,{self.num_classes}]"

        return dims

    def create_architecture_table(self) -> List[List]:
        """创建架构表格数据"""
        table_data = []

        # 标题行
        table_data.append(["模块名称", "输入维度", "输出维度", "参数量", "说明"])

        dims = self.extract_feature_dims()

        # 数据行
        rows = [
            ["视频输入", "Raw Video", dims['video_input'], "-", "RGB视频帧"],
            ["音频输入", "Raw Audio", dims['audio_input'], "-", "Mel频谱图"],
            ["", "", "", "", ""],  # 空行

            ["视频编码器\n(ViT-Base)", dims['video_input'], dims['video_backbone'], "86M", "预训练ViT"],
            ["视频投影层", dims['video_backbone'], dims['video_projection'],
             f"{self.count_parameters(self.model.video_projector) / 1e3:.1f}K" if hasattr(self.model,
                                                                                          'video_projector') else "-",
             "降维投影"],
            ["", "", "", "", ""],

            ["音频编码器\n(AST)", dims['audio_input'], dims['audio_backbone'], "86M", "预训练AST"],
            ["音频投影层", dims['audio_backbone'], dims['audio_projection'],
             f"{self.count_parameters(self.model.audio_projector) / 1e3:.1f}K" if hasattr(self.model,
                                                                                          'audio_projector') else "-",
             "降维投影"],
            ["", "", "", "", ""],

            ["** CAVA延迟估计", dims['video_projection'] + "\n" + dims['audio_projection'], dims['cava_delay'],
             f"{self.count_parameters(self.model.cava) / 1e3:.1f}K" if hasattr(self.model,
                                                                               'cava') and self.model.cava else "-",
             "估计音视频延迟"],
            ["** CAVA因果门控", dims['video_projection'] + "\n" + dims['audio_projection'], dims['cava_gate'], "-",
             "自适应对齐权重"],
            ["** CAVA对齐输出", dims['audio_projection'], dims['audio_aligned'], "-", "对齐后音频特征"],
            ["", "", "", "", ""],

            ["多模态融合\n(Transformer)", dims['video_projection'] + "\n" + dims['audio_aligned'], dims['fusion_token'],
             f"{self.count_parameters(self.model.fusion_module) / 1e3:.1f}K" if hasattr(self.model,
                                                                                        'fusion_module') else "-",
             "时序注意力融合"],
            ["", "", "", "", ""],

            ["分类器", dims['fusion_token'], dims['logits'],
             f"{self.count_parameters(self.model.classifier) / 1e3:.1f}K" if hasattr(self.model, 'classifier') else "-",
             "最终分类"],
            ["", "", "", "", ""],

            ["总计", "-", "-", f"{self.count_parameters() / 1e6:.1f}M", f"共{self.num_classes}类"],
        ]

        table_data.extend(rows)
        return table_data

    def visualize(self, output_path: str):
        """生成完整的架构可视化"""
        print("\n" + "=" * 60)
        print("🏗️  模型架构可视化")
        print("=" * 60)

        # 创建大图
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1],
                      hspace=0.35, wspace=0.3)

        # 1. 架构流程图
        ax_flow = fig.add_subplot(gs[0, :])
        self._draw_flow_diagram(ax_flow)

        # 2. 维度变化表格
        ax_table = fig.add_subplot(gs[1, :])
        self._draw_dimension_table(ax_table)

        # 3. CAVA模块详图
        ax_cava = fig.add_subplot(gs[2, 0])
        self._draw_cava_detail(ax_cava)

        # 4. 参数统计
        ax_stats = fig.add_subplot(gs[2, 1])
        self._draw_parameter_stats(ax_stats)

        plt.suptitle('Enhanced Audio-Visual Defect Detector - 模型架构',
                     fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 架构可视化已保存: {output_path}")
        print(f"   - 总参数量: {self.count_parameters() / 1e6:.1f}M")
        print(f"   - 输入: 视频{self.video_frames}帧 + 音频{self.audio_frames}帧")
        print(f"   - 输出: {self.num_classes}类")

    def _draw_flow_diagram(self, ax):
        """绘制架构流程图"""
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)
        ax.axis('off')
        ax.set_title('Model Architecture Flow', fontsize=14, fontweight='bold', pad=20)

        # 颜色方案
        c_input = '#E3F2FD'
        c_encoder = '#FFF9C4'
        c_projection = '#C8E6C9'
        c_cava = '#FFE0B2'
        c_fusion = '#F8BBD0'
        c_output = '#BBDEFB'

        # 定义模块位置 (x, y, w, h, color, text)
        boxes = [
            # 输入层
            (0.5, 4.0, 1.3, 0.9, c_input, 'Video\nInput\n[8,3,224²]'),
            (0.5, 2.0, 1.3, 0.9, c_input, 'Audio\nInput\n[8,80,201]'),

            # 编码器
            (2.3, 4.0, 1.3, 0.9, c_encoder, 'ViT\nEncoder\n[8,768]'),
            (2.3, 2.0, 1.3, 0.9, c_encoder, 'AST\nEncoder\n[8,768]'),

            # 投影层
            (4.1, 4.0, 1.3, 0.9, c_projection, 'Video\nProjection\n[8,256]'),
            (4.1, 2.0, 1.3, 0.9, c_projection, 'Audio\nProjection\n[8,256]'),

            # CAVA
            (6.2, 2.5, 1.8, 1.8, c_cava, '** CAVA\nAlignment\nDelay+Gate'),

            # 融合
            (8.8, 3.0, 1.5, 1.2, c_fusion, 'Multi-modal\nFusion\n[1,512]'),

            # 输出
            (11.0, 3.0, 1.3, 1.2, c_output, 'Classifier\n[11 classes]'),
        ]

        # 绘制模块框
        for x, y, w, h, color, text in boxes:
            box = FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.08",
                                 facecolor=color,
                                 edgecolor='#424242',
                                 linewidth=2.5)
            ax.add_patch(box)
            ax.text(x + w / 2, y + h / 2, text,
                    ha='center', va='center',
                    fontsize=8.5, fontweight='bold')

        # 绘制连接箭头
        def draw_arrow(ax, start, end, color='#212121'):
            arrow = FancyArrowPatch(start, end,
                                    arrowstyle='->',
                                    mutation_scale=22,
                                    linewidth=2.5,
                                    color=color)
            ax.add_patch(arrow)

        # 视频流
        draw_arrow(ax, (0.5 + 1.3, 4.45), (2.3, 4.45))
        draw_arrow(ax, (2.3 + 1.3, 4.45), (4.1, 4.45))
        draw_arrow(ax, (4.1 + 1.3, 4.45), (6.2, 3.8))

        # 音频流
        draw_arrow(ax, (0.5 + 1.3, 2.45), (2.3, 2.45))
        draw_arrow(ax, (2.3 + 1.3, 2.45), (4.1, 2.45))
        draw_arrow(ax, (4.1 + 1.3, 2.45), (6.2, 3.2))

        # CAVA到融合
        draw_arrow(ax, (6.2 + 1.8, 3.4), (8.8, 3.6), color='#D32F2F')

        # 融合到分类
        draw_arrow(ax, (8.8 + 1.5, 3.6), (11.0, 3.6))

        # 添加图例
        legend_elements = [
            mpatches.Patch(facecolor=c_cava, edgecolor='#424242', label='CAVA (Core Innovation)'),
            mpatches.Patch(facecolor=c_fusion, edgecolor='#424242', label='Fusion Module'),
        ]
        ax.legend(handles=legend_elements, loc='lower center',
                  ncol=2, fontsize=9, frameon=True)

    def _draw_dimension_table(self, ax):
        """绘制维度变化表格"""
        ax.axis('off')
        ax.set_title('Feature Dimension Transition Table',
                     fontsize=14, fontweight='bold', pad=20)

        # 获取表格数据
        table_data = self.create_architecture_table()

        # 创建表格
        table = ax.table(cellText=table_data,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.02, 0.05, 0.96, 0.9])

        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 2.2)

        # 设置样式
        for i in range(len(table_data)):
            for j in range(5):
                cell = table[(i, j)]

                if i == 0:  # 标题行
                    cell.set_facecolor('#1565C0')
                    cell.set_text_props(weight='bold', color='white', fontsize=9.5)
                    cell.set_height(0.08)
                elif table_data[i][0] == "":  # 空行
                    cell.set_facecolor('#F5F5F5')
                    cell.set_height(0.03)
                elif '**' in str(table_data[i][0]):  # CAVA行
                    cell.set_facecolor('#FFF59D')
                    cell.set_text_props(fontsize=8.5)
                elif table_data[i][0] == "总计":  # 总计行
                    cell.set_facecolor('#E0E0E0')
                    cell.set_text_props(weight='bold', fontsize=9)
                elif i % 2 == 0:
                    cell.set_facecolor('#FAFAFA')
                    cell.set_text_props(fontsize=8)
                else:
                    cell.set_facecolor('#FFFFFF')
                    cell.set_text_props(fontsize=8)

                cell.set_edgecolor('#BDBDBD')
                cell.set_linewidth(1)

    def _draw_cava_detail(self, ax):
        """绘制CAVA模块详细结构"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_title('CAVA Module Detail', fontsize=12, fontweight='bold', pad=15)

        # 输入
        box_v = FancyBboxPatch((0.5, 5.5), 2, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#BBDEFB',
                               edgecolor='#1976D2', linewidth=2.5)
        ax.add_patch(box_v)
        ax.text(1.5, 6.1, 'Video Feat\nV[T,D]',
                ha='center', va='center', fontsize=9.5, fontweight='bold')

        box_a = FancyBboxPatch((0.5, 2.5), 2, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#C8E6C9',
                               edgecolor='#388E3C', linewidth=2.5)
        ax.add_patch(box_a)
        ax.text(1.5, 3.1, 'Audio Feat\nA[T,D]',
                ha='center', va='center', fontsize=9.5, fontweight='bold')

        # 延迟估计器
        box_delay = FancyBboxPatch((3.5, 4), 2, 1.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#FFE082',
                                   edgecolor='#F57C00', linewidth=2.5)
        ax.add_patch(box_delay)
        ax.text(4.5, 4.9, 'Delay\nEstimator\nδ ∈ [0,6]',
                ha='center', va='center', fontsize=9.5, fontweight='bold')

        # 因果门控
        box_gate = FancyBboxPatch((6.5, 4), 2, 1.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#FFAB91',
                                  edgecolor='#E64A19', linewidth=2.5)
        ax.add_patch(box_gate)
        ax.text(7.5, 4.9, 'Causal\nGate\ng(t) ∈ [0,1]',
                ha='center', va='center', fontsize=9.5, fontweight='bold')

        # 对齐输出
        box_out = FancyBboxPatch((6.5, 1.5), 2, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#A5D6A7',
                                 edgecolor='#2E7D32', linewidth=2.5)
        ax.add_patch(box_out)
        ax.text(7.5, 2.1, "Aligned A'\nA'[T,D]",
                ha='center', va='center', fontsize=9.5, fontweight='bold')

        # 箭头
        def draw_arrow_detail(start, end):
            arrow = FancyArrowPatch(start, end,
                                    arrowstyle='->',
                                    mutation_scale=18,
                                    linewidth=2.2,
                                    color='#212121')
            ax.add_patch(arrow)

        draw_arrow_detail((2.5, 6.1), (3.5, 4.9))
        draw_arrow_detail((2.5, 3.1), (3.5, 4.9))
        draw_arrow_detail((5.5, 4.9), (6.5, 4.9))
        draw_arrow_detail((7.5, 4), (7.5, 2.7))

        # 公式框
        formula_box = FancyBboxPatch((2, 0.3), 6, 0.7,
                                     boxstyle="round,pad=0.08",
                                     facecolor='#FFF9C4',
                                     edgecolor='#F57F17',
                                     linewidth=2)
        ax.add_patch(formula_box)
        ax.text(5, 0.67, "A'(t) = g(t) * Shift(A(t), delta)",
                fontsize=11.5, ha='center', va='center', weight='bold',
                family='monospace')

    def _draw_parameter_stats(self, ax):
        """绘制参数统计饼图"""
        ax.set_title('Parameter Statistics', fontsize=12, fontweight='bold', pad=15)

        # 统计各部分参数
        total_params = self.count_parameters()

        params_data = []
        labels_data = []
        colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFD700', '#FF99CC', '#99CCFF']

        # 视频编码器
        params_data.append(86e6)
        labels_data.append('Video\nEncoder\n86.0M')

        # 音频编码器
        params_data.append(86e6)
        labels_data.append('Audio\nEncoder\n86.0M')

        # 投影层
        proj_params = 0
        if hasattr(self.model, 'video_projector'):
            proj_params += self.count_parameters(self.model.video_projector)
        if hasattr(self.model, 'audio_projector'):
            proj_params += self.count_parameters(self.model.audio_projector)
        if proj_params > 0:
            params_data.append(proj_params)
            labels_data.append(f'Projectors\n{proj_params / 1e6:.1f}M')

        # CAVA
        if hasattr(self.model, 'cava') and self.model.cava:
            cava_params = self.count_parameters(self.model.cava)
            if cava_params > 0:
                params_data.append(cava_params)
                labels_data.append(f'** CAVA\n{cava_params / 1e3:.1f}K')

        # 融合
        if hasattr(self.model, 'fusion_module'):
            fusion_params = self.count_parameters(self.model.fusion_module)
            if fusion_params > 0:
                params_data.append(fusion_params)
                labels_data.append(f'Fusion\n{fusion_params / 1e6:.1f}M')

        # 分类器
        if hasattr(self.model, 'classifier'):
            cls_params = self.count_parameters(self.model.classifier)
            if cls_params > 0:
                params_data.append(cls_params)
                labels_data.append(f'Classifier\n{cls_params / 1e3:.1f}K')

        # 绘制饼图
        wedges, texts, autotexts = ax.pie(
            params_data,
            labels=labels_data,
            colors=colors[:len(params_data)],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 8.5, 'weight': 'bold'},
            pctdistance=0.85
        )

        # 美化百分比文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8)
            autotext.set_weight('bold')

        # 总参数信息
        ax.text(0, -1.45, f'Total Parameters: {total_params / 1e6:.1f}M',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#A5D6A7',
                          edgecolor='#2E7D32', linewidth=2))

        # 可训练信息
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total_params - trainable

        info_text = f'Trainable: {trainable / 1e6:.1f}M ({trainable / total_params * 100:.1f}%)\n'
        info_text += f'Frozen: {frozen / 1e6:.1f}M ({frozen / total_params * 100:.1f}%)'

        ax.text(0, -1.85, info_text,
                ha='center', fontsize=8.5, style='italic')


def main():
    parser = argparse.ArgumentParser(description='Model Architecture Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path')
    parser.add_argument('--output', type=str, default='./model_architecture.png',
                        help='Output image path')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🏗️  Model Architecture Visualization Tool")
    print("=" * 60)

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")

    # Load model
    print(f"📦 Loading model: {args.checkpoint}")
    model_cfg = cfg.get("model", {})
    model_cfg["num_classes"] = cfg["data"]["num_classes"]

    model = EnhancedAVTopDetector({
        "model": model_cfg,
        "fusion": model_cfg.get("fusion", {}),
        "cava": cfg.get("cava", {})
    }).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"✅ Model loaded successfully")

    # Create visualizer
    visualizer = ArchitectureVisualizer(model, cfg)

    # Generate visualization
    visualizer.visualize(args.output)

    print("\n" + "=" * 60)
    print("🎉 Architecture visualization completed!")
    print(f"📁 Output: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()