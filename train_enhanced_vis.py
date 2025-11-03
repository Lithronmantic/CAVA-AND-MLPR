#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, yaml, random, numpy as np, torch
import multiprocessing as mp
from pathlib import Path

# 固定唯一来源：只引用，不自定义
from dataset import AVFromCSV, safe_collate_fn
from strong_trainer import StrongTrainer

# --- Robust batch unpack ---
def _unpack_batch(b):
    """
    支持以下格式：
      - tuple/list: (v, a, y) 或 (v, a, y, ids/meta)
      - dict: {'video':..., 'audio':..., 'label':..., 'ids':...}（键名大小写不敏感）
    返回: (v, a, y, ids_or_meta)；若无 ids/meta 则为 None
    """
    if isinstance(b, dict):
        # 尝试常见键（大小写不敏感）
        keys = {k.lower(): k for k in b.keys()}
        def _req(name):
            if name not in keys:
                raise KeyError(f"batch dict 缺少必须键: {name}")
            return b[keys[name]]
        v = _req('video')
        a = _req('audio')
        y = _req('label')
        ids = b.get(keys.get('ids')) if 'ids' in keys else (b.get(keys.get('meta')) if 'meta' in keys else None)
        return v, a, y, ids

    if isinstance(b, (list, tuple)):
        if len(b) >= 3:
            v, a, y = b[:3]
            ids = b[3] if len(b) >= 4 else None
            return v, a, y, ids

    raise ValueError(f"Unsupported batch structure: type={type(b)}, len={len(b) if hasattr(b,'__len__') else 'N/A'}")


def main():
    parser = argparse.ArgumentParser(description='🚀 Clean Training Entry')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/run")
    parser.add_argument("--diagnose", action="store_true",
                        help="仅做构建/首批次/前向诊断，不进入训练")
    args = parser.parse_args()

    # 强制把 YAML 解析成 dict
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"配置文件解析结果不是 dict，请检查 YAML：{args.config}")

    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print("\n" + "="*80)
    print("🚀 Enhanced Semi-Supervised Training Script (CLEAN ENTRY)")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Seed:   {seed}")
    print("="*80)

    if args.diagnose:
        # 构建 Trainer（内部会构建 dataloader / model / optimizer 等）
        st = StrongTrainer(cfg, args.output)
        print("[DIAG] Building one train batch and forward ...")
        st.model.eval()
        try:
            it = iter(st.loader_l)
            b = next(it)
        except StopIteration:
            raise RuntimeError("训练集为空，无法诊断。请检查 labeled_csv 或数据过滤条件。")

        with torch.no_grad():
            # ✅ 使用健壮解包：兼容 3/4 元组和 dict
            v, a, y, _ = _unpack_batch(b)
            # 标签维度：若是 one-hot [B,C]，转成 index [B]
            if hasattr(y, "ndim") and y.ndim == 2:
                y = y.argmax(dim=1)
            v, a = v.to(st.device), a.to(st.device)

            # 前向
            out = st._forward(v, a)
            if out is None:
                raise RuntimeError("前向返回 None，请检查模型/输入。")
            logits = out["clip_logits"] if isinstance(out, dict) and "clip_logits" in out else out

            # 额外打印关键形状，便于快速定位问题
            vshape = tuple(v.shape) if hasattr(v, "shape") else type(v)
            ashape = tuple(a.shape) if hasattr(a, "shape") else type(a)
            yshape = tuple(y.shape) if hasattr(y, "shape") else type(y)
            lshape = tuple(logits.shape) if hasattr(logits, "shape") else type(logits)
            print(f"[DIAG] batch: v={vshape}, a={ashape}, y={yshape}")
            print(f"[DIAG] model forward OK, logits shape={lshape}")

            # 如有 CAVA，打印关键辅助量可用性
            if isinstance(out, dict):
                flags = {k: (out.get(k) is not None) for k in
                         ["audio_seq", "audio_aligned", "video_proj", "causal_gate",
                          "delay_frames", "causal_prob", "causal_prob_dist", "pred_delay"]}
                print(f"[DIAG] CAVA keys: {flags}")

        print("[DIAG] Done.")
        return

    print("➡️ 使用 StrongTrainer（不平衡强化 + 可选分组学习率/AMP）")
    try:
        st = StrongTrainer(cfg, args.output)
        st.train()
    except Exception:
        # 打印完整堆栈，避免只看到“Traceback”但无细节
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "="*80)
    print("✅ All Done!")
    print("="*80)
    print(f"📂 Results:     {args.output}")
    print(f"💾 Checkpoints: {args.output}/checkpoints/")
    print(f"📝 Logs:        {args.output}/logs/ (按需追加)")


if __name__ == "__main__":
    # Windows 安全入口
    mp.freeze_support()
    main()
