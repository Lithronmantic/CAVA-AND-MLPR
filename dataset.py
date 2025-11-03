# -*- coding: utf-8 -*-
import os, csv
from typing import List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# -------------------- 小工具 --------------------
def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _to_none_like(v):
    """将 '', 'none', 'null' 规范成 None。"""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null"):
        return None
    return v

def resolve_path(rel: str, data_root: Optional[str]) -> Path:
    """
    鲁棒路径解析：
    - 支持绝对路径直接返回
    - 支持 data_root=None/'none'：不拼接
    - 自动去除 CSV 里多余的前缀（data/, ./, .\）
    - 展开环境变量与 ~
    """
    s = (rel or "").strip().strip('"').strip("'")
    s = os.path.expandvars(os.path.expanduser(s))
    p = Path(s)
    # 已是绝对路径且存在
    if p.is_absolute() and p.exists():
        return p

    # 需要拼 data_root
    root = _to_none_like(data_root)
    if root is not None:
        base = Path(os.path.expandvars(os.path.expanduser(str(root))))
        cand = base / s
        if cand.exists():
            return cand
        # 去掉可能多写的前缀再拼
        for prefix in ("data/", "data\\", "./", ".\\"):
            if s.startswith(prefix):
                cand2 = base / s[len(prefix):]
                if cand2.exists():
                    return cand2

    # 最后尝试相对路径（不一定存在，让上游决定）
    return Path(s)

# -------------------- 数据集 --------------------
class AVFromCSV(Dataset):
    """
    单标签多类（与 CrossEntropy/FocalCE 对齐）的音视频数据集。

    支持 CSV 列（至少满足其一）：
      - label:       整数索引 [0..C-1]
      - class_name:  类名（必须能在 class_names 中找到）
      - class_0..class_{C-1}: one-hot（恰有一个 1）

    建议/可选列（用于时域对齐）：
      - video_start_frame, video_end_frame
      - audio_start_s,    audio_end_s

    返回：
      video: [T, 3, H, W], float32 in [0,1]
      audio: [T, mel_bins, frames_per_slice]
      y:     torch.long（单标签类索引）
    """
    def __init__(
        self,
        csv_path: str,
        data_root: Optional[str],
        num_classes: int,
        class_names: List[str],
        video_cfg: Dict,
        audio_cfg: Dict,
        is_unlabeled: bool = False,
        augmentation_mode: Optional[str] = None,  # 保留接口：'weak'/'strong'/None
    ):
        super().__init__()
        # root 为 None 表示不加任何前缀（与自检脚本一致）
        self.root: Optional[str] = _to_none_like(data_root)
        self.C = int(num_classes)
        self.class_names = list(class_names)

        # video
        self.T_v = int(video_cfg.get("num_frames", video_cfg.get("frames", 8)))
        v_size = int(video_cfg.get("size", 224))
        self.v_size = (v_size, v_size)

        # audio 基本参数
        self.sr = int(audio_cfg.get("sample_rate", audio_cfg.get("sr", 16000)))
        self.mel_bins = int(audio_cfg.get("n_mels", audio_cfg.get("mel_bins", 128)))
        self.frames_per_slice = int(audio_cfg.get("segment_frames", audio_cfg.get("target_len", 64)))

        # STFT 关键参数（与预处理一致，防止不同库默认不一致）
        self.n_fft = int(audio_cfg.get("n_fft", 2048))
        self.hop_length = int(audio_cfg.get("hop_length", 512))
        # 我们在内部做了长度兜底，因此默认 center=False 更稳定
        self.center = bool(audio_cfg.get("center", False))
        self.pad_mode = str(audio_cfg.get("pad_mode", "reflect"))

        self.is_unlabeled = bool(is_unlabeled)
        self.rows: List[Dict] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_v = row.get("video_path", row.get("video", "")) or ""
                raw_a = row.get("audio_path", row.get("audio", "")) or ""

                vpath = resolve_path(raw_v, self.root)
                apath = resolve_path(raw_a, self.root)

                y_idx = None
                if not self.is_unlabeled:
                    y_idx = self._parse_label(row)

                self.rows.append({
                    "clip_id": row.get("sample", os.path.basename(str(vpath)) or os.path.basename(str(apath)) or str(len(self.rows))),
                    "video_path": str(vpath),
                    "audio_path": str(apath),
                    "label_idx": y_idx,
                    "video_start_frame": _safe_int(row.get("video_start_frame")),
                    "video_end_frame":   _safe_int(row.get("video_end_frame")),
                    "audio_start_s":     _safe_float(row.get("audio_start_s")),
                    "audio_end_s":       _safe_float(row.get("audio_end_s")),
                })

    # -------- 标签解析：锁死为 Long 索引 --------
    def _parse_label(self, row: Dict) -> int:
        v = str(row.get("label", "")).strip()
        if v != "" and v.lstrip("-").isdigit():
            idx = int(v)
            if not (0 <= idx < self.C):
                raise ValueError(f"label 超范围: {idx}/{self.C}")
            return idx

        cname = (row.get("class_name") or "").strip()
        if cname:
            if cname in self.class_names:
                return self.class_names.index(cname)
            else:
                raise ValueError(f"class_name 不在 class_names: {cname}")

        # one-hot
        oh = []
        for i in range(self.C):
            k = f"class_{i}"
            try:
                oh.append(float(row.get(k, 0)))
            except Exception:
                oh.append(0.0)
        if sum(x > 0.5 for x in oh) == 1:
            return int(np.argmax(oh))

        raise ValueError(f"无法解析标签（需要 label / class_name / one-hot）：{row}")

    def __len__(self):
        return len(self.rows)

    # -------- 视频加载：按起止帧均匀采样 T_v 帧 --------
    def _load_video_frames(self, path: str, start_f: Optional[int], end_f: Optional[int]) -> torch.Tensor:
        import cv2
        H, W = self.v_size[1], self.v_size[0]
        T = self.T_v

        if not path or (not os.path.exists(path)):
            # 视频缺失时返回随机帧（保持训练不中断；音频是关键模态）
            frames = np.random.rand(T, H, W, 3).astype("float32")
            return torch.from_numpy(frames).permute(0,3,1,2).contiguous()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            frames = np.random.rand(T, H, W, 3).astype("float32")
            return torch.from_numpy(frames).permute(0,3,1,2).contiguous()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        s = 0 if (start_f is None or start_f < 0) else min(start_f, total-1)
        e = total-1 if (end_f   is None or end_f   < 0) else min(end_f,   total-1)
        if e < s:
            e = s

        idx = np.linspace(s, e, num=T).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = (np.random.rand(H, W, 3)*255).astype("uint8")
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            frame = frame[:, :, ::-1]  # BGR -> RGB
            frames.append(frame)
        cap.release()

        frames = np.stack(frames, axis=0).astype("float32") / 255.0  # [T,H,W,3]
        frames = torch.from_numpy(frames).permute(0,3,1,2).contiguous()  # [T,3,H,W]
        return frames

    # -------- 内部：将 numpy 波形 -> log-mel（长度兜底，防告警） --------
    def _compute_mel_from_numpy(self, y: np.ndarray, sr: int) -> torch.Tensor:
        """
        - 空数组直接报错
        - 长度 < n_fft 则两侧或尾部补齐到 n_fft
        - 统一 center=False，避免 librosa 在短序列时的 STFT 告警
        """
        if y is None or (hasattr(y, "size") and y.size == 0):
            raise ValueError("Empty waveform before STFT/mel.")

        if y.ndim > 1:
            y = np.mean(y, axis=-1)  # 转单通道

        if len(y) < self.n_fft:
            pad = self.n_fft - len(y)
            # 尾部补零（也可用 'reflect'，但零填充在工业噪声下更稳）
            y = np.pad(y, (0, pad), mode="constant", constant_values=0.0)

        try:
            import librosa
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=self.mel_bins, center=False, power=2.0
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            mel = torch.from_numpy(S_db).unsqueeze(0).float()  # [1, mel, Tspec]
            return mel
        except Exception as e:
            # 极端情况下，给出可训练的零张量（并附加最少噪声，避免 -inf）
            Tspec_min = max(1, (len(y) - self.n_fft) // max(1, self.hop_length) + 1)
            mel = torch.zeros(1, self.mel_bins, Tspec_min, dtype=torch.float32)
            return mel

    # -------- 音频加载：裁段→logMel→重采样到 T*frames_per_slice → reshape --------
    def _load_audio_slices(self, path: str, start_s: Optional[float], end_s: Optional[float]) -> torch.Tensor:
        T = self.T_v
        mel_bins = self.mel_bins
        fps = self.frames_per_slice

        p = Path(path)
        if not p.exists():
            # 路径错误不再“静默随机”，直接报错以便早发现
            raise FileNotFoundError(f"Audio not found: {p}")

        # 先尝试 torchaudio 读取（速度快，解码稳定）
        mel = None
        try:
            import torchaudio
            wav, s = torchaudio.load(str(p))  # [C, N]
            if s != self.sr:
                wav = torchaudio.functional.resample(wav, s, self.sr)
                s = self.sr
            mono = wav.mean(dim=0)  # [N]

            # 切片（单位：秒）
            if start_s is not None and end_s is not None and end_s > start_s:
                s_i = int(max(0.0, start_s) * s)
                e_i = int(end_s * s)
                e_i = max(e_i, s_i + 1)
                mono = mono[s_i:e_i]

            # 防止长度过短导致 STFT 报警：补到 n_fft
            if mono.numel() < self.n_fft:
                pad = self.n_fft - mono.numel()
                mono = torch.nn.functional.pad(mono, (0, pad), value=0.0)

            # 构造与 librosa 对齐的 Mel 参数
            mel_extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=self.n_fft, win_length=self.n_fft,
                hop_length=self.hop_length, n_mels=mel_bins, center=False,
                pad_mode=self.pad_mode, power=2.0
            )
            mel = mel_extractor(mono.unsqueeze(0))  # [1, mel, Tspec]
            mel = torch.log(mel + 1e-6)
        except Exception:
            # 回退到 librosa（并统一参数）
            try:
                import librosa
                y, s = librosa.load(str(p), sr=self.sr, mono=True)
                if start_s is not None and end_s is not None and end_s > start_s:
                    s_i = int(max(0.0, start_s) * s)
                    e_i = int(end_s * s)
                    e_i = max(e_i, s_i + 1)
                    y = y[s_i:e_i]
                mel = self._compute_mel_from_numpy(y, s)  # [1, mel, Tspec]
            except Exception as e:
                # 最后的兜底：给出合理形状的零张量，避免训练中断
                mel = torch.zeros(1, mel_bins, 1, dtype=torch.float32)

        # 统一对齐到 T * fps
        Tspec = int(mel.size(-1))
        target_T = int(T * fps)
        if Tspec == 0:
            mel = mel.new_zeros(1, mel_bins, target_T)
        elif Tspec >= target_T:
            idx = torch.linspace(0, Tspec - 1, steps=target_T).long()
            mel = mel.index_select(-1, idx)
        else:
            pad = target_T - Tspec
            mel = torch.cat([mel, mel.new_zeros(1, mel_bins, pad)], dim=-1)

        # [1, mel, T*fps] -> [T, mel, fps]
        mel = mel.view(1, mel_bins, T, fps).permute(2, 1, 3, 0).squeeze(-1).contiguous()
        return mel  # [T, mel_bins, fps]

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        v = self._load_video_frames(row["video_path"], row["video_start_frame"], row["video_end_frame"])
        a = self._load_audio_slices(row["audio_path"], row["audio_start_s"], row["audio_end_s"])

        if self.is_unlabeled:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = torch.tensor(int(row["label_idx"]), dtype=torch.long)

        # 新增：返回一个稳定的“样本唯一ID”。这里直接用索引 idx（int），也可以换成 row["clip_id"] 的哈希。
        ids = torch.tensor(idx, dtype=torch.long)
        return v, a, y, ids

# 旧版（保留兼容）：如果你的老代码只接收 3 元组，可以继续用这个
def safe_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    # 兼容 3/4 元组
    if len(batch) == 0:
        return None
    if len(batch[0]) == 3:
        v = torch.stack([b[0] for b in batch], dim=0)
        a = torch.stack([b[1] for b in batch], dim=0)
        y = torch.stack([b[2] for b in batch], dim=0).long()
        return v, a, y
    else:
        v = torch.stack([b[0] for b in batch], dim=0)
        a = torch.stack([b[1] for b in batch], dim=0)
        y = torch.stack([b[2] for b in batch], dim=0).long()
        ids = torch.stack([b[3] for b in batch], dim=0).long()
        return v, a, y, ids
# 新版（推荐用于无标签 DataLoader）：明确返回 ids
# 新版（推荐用于无标签 DataLoader）：明确返回 ids
def safe_collate_fn_with_ids(batch):
    batch = [b for b in batch if b is not None]
    v = torch.stack([b[0] for b in batch], dim=0)          # [B,T,3,H,W]
    a = torch.stack([b[1] for b in batch], dim=0)          # [B,T,mel,f]
    y = torch.stack([b[2] for b in batch], dim=0).long()   # [B]
    ids = torch.stack([b[3] for b in batch], dim=0).long() # [B]
    return v, a, y, ids
