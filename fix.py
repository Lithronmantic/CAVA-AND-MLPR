#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证修复脚本 - 检查所有修复是否正确应用

运行此脚本来验证：
1. 训练器类是否包含必要的修复方法
2. 配置文件参数是否正确更新
3. 关键依赖是否可用

使用方法：
    python verify_fix.py [--trainer strong_trainer.py] [--config selfsup_sota.yaml]
"""

import sys
import argparse
from pathlib import Path
import yaml
import importlib.util


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")


def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")


def load_module_from_file(module_name, file_path):
    """从文件路径加载Python模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print_error(f"加载模块失败: {e}")
        return None


def verify_trainer_file(trainer_path):
    """验证训练器文件的修复"""
    print_header("检查 1: 验证训练器文件")

    if not trainer_path.exists():
        print_error(f"训练器文件不存在: {trainer_path}")
        return False

    print_info(f"检查文件: {trainer_path}")

    # 读取文件内容
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []
    checks = {
        "_reset_scaler_if_needed": False,
        "_reset_scaler_calls": 0,
        "allow_unused=True": False,
        "grads[idx] is not None": False,
    }

    # 检查1: _reset_scaler_if_needed方法是否存在
    if "def _reset_scaler_if_needed(self):" in content:
        checks["_reset_scaler_if_needed"] = True
        print_success("找到 _reset_scaler_if_needed 方法")
    else:
        issues.append("缺少 _reset_scaler_if_needed 方法")
        print_error("缺少 _reset_scaler_if_needed 方法")

    # 检查2: _reset_scaler_if_needed是否被调用
    checks["_reset_scaler_calls"] = content.count("self._reset_scaler_if_needed()")
    if checks["_reset_scaler_calls"] >= 5:
        print_success(f"找到 {checks['_reset_scaler_calls']} 处 _reset_scaler_if_needed 调用")
    else:
        issues.append(f"_reset_scaler_if_needed 调用次数过少 ({checks['_reset_scaler_calls']}/推荐≥5)")
        print_warning(f"_reset_scaler_if_needed 调用次数: {checks['_reset_scaler_calls']} (推荐≥5)")

    # 检查3: allow_unused=True
    if "allow_unused=True" in content:
        checks["allow_unused=True"] = True
        print_success("找到 allow_unused=True 参数")
    else:
        issues.append("缺少 allow_unused=True 参数")
        print_warning("未找到 allow_unused=True 参数（如果不使用元学习可忽略）")

    # 检查4: 梯度None检查
    if "grads[idx] is not None" in content or "if grads[idx]:" in content:
        checks["grads[idx] is not None"] = True
        print_success("找到梯度None检查")
    else:
        print_warning("未找到梯度None检查（如果不使用元学习可忽略）")

    # 额外检查：关键修复点
    print("\n额外检查：")

    # 检查异常处理后是否有重置
    continue_without_reset = 0
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'continue' in line and i > 0:
            # 检查前5行是否有reset调用
            prev_lines = '\n'.join(lines[max(0, i - 5):i])
            if '_reset_scaler_if_needed' not in prev_lines and 'nan_count' in prev_lines:
                continue_without_reset += 1

    if continue_without_reset > 0:
        print_warning(f"发现 {continue_without_reset} 处可能缺少scaler重置的continue")
    else:
        print_success("所有continue语句前都有适当处理")

    # 总结
    print(f"\n{'-' * 60}")
    if len(issues) == 0:
        print_success("训练器文件检查完全通过！")
        return True
    else:
        print_error(f"发现 {len(issues)} 个问题：")
        for issue in issues:
            print(f"  - {issue}")
        return False


def verify_config_file(config_path):
    """验证配置文件的修复"""
    print_header("检查 2: 验证配置文件")

    if not config_path.exists():
        print_error(f"配置文件不存在: {config_path}")
        return False

    print_info(f"检查文件: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print_error(f"加载配置文件失败: {e}")
        return False

    issues = []
    recommendations = {
        "cava.lambda_align": (0.02, 0.03, "<="),
        "cava.lambda_edge": (0.005, 0.01, "<="),
        "mlpr.ema_decay": (0.9995, 0.999, ">="),
        "mlpr.meta_lr": (5e-5, 1e-4, "<="),
        "mlpr.lambda_u": (0.3, 0.5, "<="),
        "mlpr.meta_interval": (20, 10, ">="),
        "training.learning_rate": (1e-5, 5e-5, "<="),
        "training.gradient_clip": (0.5, 1.0, "<="),
    }

    def get_nested_value(d, path):
        """获取嵌套字典的值"""
        keys = path.split('.')
        val = d
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return None
        return val

    print("\n关键参数检查：")
    for param_path, (recommended, original, op) in recommendations.items():
        current = get_nested_value(cfg, param_path)

        if current is None:
            print_warning(f"{param_path}: 未找到")
            continue

        # 检查是否符合推荐
        if op == "<=":
            is_good = current <= recommended
        else:  # ">="
            is_good = current >= recommended

        if is_good:
            print_success(f"{param_path}: {current} (推荐 {op} {recommended})")
        else:
            print_warning(f"{param_path}: {current} (推荐 {op} {recommended}, 原值: {original})")
            issues.append(f"{param_path} 未优化")

    # 检查AMP设置
    amp_enabled = get_nested_value(cfg, "training.amp")
    if amp_enabled is True:
        print_success("training.amp: 已启用（配合修复的scaler使用）")
    else:
        print_warning("training.amp: 未启用或禁用（可能降低训练速度）")

    # 总结
    print(f"\n{'-' * 60}")
    if len(issues) == 0:
        print_success("配置文件检查完全通过！")
        return True
    else:
        print_warning(f"配置文件有 {len(issues)} 处可优化：")
        for issue in issues:
            print(f"  - {issue}")
        print_info("这些参数不是必需的，但推荐优化以提高稳定性")
        return True  # 配置问题不算严重错误


def verify_dependencies():
    """验证关键依赖"""
    print_header("检查 3: 验证依赖包")

    required = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
        "sklearn": "scikit-learn",
    }

    optional = {
        "torchaudio": "音频处理",
        "cv2": "OpenCV (视频处理)",
        "librosa": "音频处理备选",
    }

    all_ok = True

    print("必需依赖：")
    for module, name in required.items():
        try:
            __import__(module)
            print_success(f"{name} ({module})")
        except ImportError:
            print_error(f"{name} ({module}) - 缺失！")
            all_ok = False

    print("\n可选依赖：")
    for module, name in optional.items():
        try:
            __import__(module)
            print_success(f"{name} ({module})")
        except ImportError:
            print_warning(f"{name} ({module}) - 未安装")

    # 检查PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print_success(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            print_info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print_warning("CUDA不可用（将使用CPU训练，速度较慢）")
    except Exception as e:
        print_error(f"检查PyTorch失败: {e}")

    print(f"\n{'-' * 60}")
    if all_ok:
        print_success("依赖检查通过！")
    else:
        print_error("部分依赖缺失，请安装后重试")

    return all_ok


def verify_data_files(config_path):
    """验证数据文件是否存在"""
    print_header("检查 4: 验证数据文件")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except:
        print_warning("无法加载配置文件，跳过数据文件检查")
        return True

    data_cfg = cfg.get("data", {})
    files_to_check = {
        "labeled_csv": "标注训练集",
        "val_csv": "验证集",
        "unlabeled_csv": "无标注数据集",
    }

    all_ok = True
    for key, name in files_to_check.items():
        filepath = data_cfg.get(key)
        if not filepath:
            continue

        path = Path(filepath)
        if path.exists():
            print_success(f"{name}: {filepath}")
        else:
            print_warning(f"{name}: {filepath} - 文件不存在")
            all_ok = False

    print(f"\n{'-' * 60}")
    if all_ok:
        print_success("数据文件检查通过！")
    else:
        print_warning("部分数据文件不存在，请检查路径配置")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description='验证梯度异常修复是否正确应用')
    parser.add_argument('--trainer', type=str, default='strong_trainer.py',
                        help='训练器文件路径')
    parser.add_argument('--config', type=str, default='selfsup_sota.yaml',
                        help='配置文件路径')
    parser.add_argument('--skip-data', action='store_true',
                        help='跳过数据文件检查')

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          梯度异常修复验证脚本 v2.0                         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")

    trainer_path = Path(args.trainer)
    config_path = Path(args.config)

    results = {
        "trainer": False,
        "config": False,
        "dependencies": False,
        "data": True,  # 默认通过，可选
    }

    # 执行检查
    results["trainer"] = verify_trainer_file(trainer_path)
    results["config"] = verify_config_file(config_path)
    results["dependencies"] = verify_dependencies()

    if not args.skip_data:
        results["data"] = verify_data_files(config_path)

    # 最终总结
    print_header("最终检查结果")

    all_passed = all(results.values())
    critical_passed = results["trainer"] and results["dependencies"]

    for check, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{check.capitalize():20s}: {status}")

    print(f"\n{'-' * 60}\n")

    if all_passed:
        print_success("🎉 所有检查通过！修复已正确应用！")
        print_info("\n可以开始训练：")
        print(f"    python train.py --config {args.config} --output ./runs/fixed_exp")
        return 0
    elif critical_passed:
        print_warning("⚠️  核心修复已应用，但有些配置可以优化")
        print_info("\n可以开始训练，但建议先查看上面的警告")
        print(f"    python train.py --config {args.config} --output ./runs/fixed_exp")
        return 0
    else:
        print_error("❌ 检查失败！请先修复上述问题")
        print_info("\n请参考以下文档：")
        print("    - QUICK_FIX_CHECKLIST.md : 快速修复指南")
        print("    - FIX_REPORT.md : 详细修复方案")
        return 1


if __name__ == "__main__":
    sys.exit(main())