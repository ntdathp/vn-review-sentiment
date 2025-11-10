"""
PhoBERT wrapper for Chat Toolbox GUI — FORCE CPU version.

- Luôn chạy trên CPU (bỏ qua CUDA hoàn toàn).
- Nạp model với torch_dtype=float32, low_cpu_mem_usage=True, device_map={"": "cpu"}.
- API:
    * infer(text: str) -> str
    * set_config(**kwargs)  # thay đổi tham số runtime: max_len, use_seg, ...
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== BẮT BUỘC CPU: chặn mọi truy cập GPU ngay từ đầu =====
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ===== Text pipeline (giống test_phobert.py) =====
from textproc import (
    LABELS_5, normalize_text, maybe_segment, sentiment_prefix,
    approx_diacritic_ratio, restore_diacritics
)

# ===== Model dir autodetect =====
_DEF_CANDIDATES: List[str] = [
    os.environ.get("PHOBERT_MODEL_DIR", ""),     
    "/home/dat/llm_ws/phobert/phobert_5cls_clean",           # path 1
]
MODEL_DIR_PHOBERT = next((p for p in _DEF_CANDIDATES if p and os.path.isdir(p)), "")

# ===== Labels =====
IDX2LABEL: Dict[int, str] = {i: l for i, l in enumerate(LABELS_5)}

# ===== Runtime config =====
CFG = {
    "max_len": 160,
    "use_seg": False,
    "use_normalize": True,
    "use_prefix": True,
    "auto_restore": True,
    "restore_threshold": 0.03,
    "neutral_penalty": 0.0,
    "cpu_threads": 0,   # 0 = mặc định PyTorch; có thể chỉnh bằng set_config(cpu_threads=4)
}

# ===== Device & Handles (ép CPU) =====
_DEVICE = torch.device("cpu")
if CFG.get("cpu_threads", 0) and CFG["cpu_threads"] > 0:
    torch.set_num_threads(int(CFG["cpu_threads"]))

_tok: Optional[AutoTokenizer] = None
_mdl: Optional[AutoModelForSequenceClassification] = None


def set_config(**kwargs: Any):
    """Update runtime config, e.g., set_config(use_seg=True, neutral_penalty=-0.05, cpu_threads=4)."""
    global _DEVICE, _mdl
    for k, v in kwargs.items():
        if k in CFG:
            CFG[k] = v
    # Nếu đổi số luồng CPU sau khi đã import
    if "cpu_threads" in kwargs:
        ct = int(CFG["cpu_threads"])
        if ct > 0:
            torch.set_num_threads(ct)
    # Nếu model đã nạp, đảm bảo nó ở CPU (an toàn)
    if _mdl is not None:
        _mdl.to(_DEVICE)


def _ensure():
    """Load tokenizer & model một lần; luôn ở CPU."""
    global _tok, _mdl, MODEL_DIR_PHOBERT
    if _tok is not None and _mdl is not None:
        return

    # Tìm checkpoint
    if not MODEL_DIR_PHOBERT:
        for p in _DEF_CANDIDATES:
            if p and os.path.isdir(p):
                MODEL_DIR_PHOBERT = p
                break
    if not MODEL_DIR_PHOBERT or not os.path.isdir(MODEL_DIR_PHOBERT):
        raise FileNotFoundError(
            "Checkpoint not found. Set PHOBERT_MODEL_DIR hoặc tạo một trong các thư mục:\n"
            "  - /home/dat/llm_ws/phobert/phobert_5cls_clean\n"
            "  - /home/dat/llm_ws/phobert_5cls_clean"
        )

    # Tokenizer (parity với test_phobert: use_fast=False)
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR_PHOBERT, use_fast=False)

    # Model: nạp CPU-friendly
    _mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR_PHOBERT,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},   # tránh tự map sang CUDA
    )
    _mdl.to(_DEVICE).eval()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, keepdims=True)


def _preprocess(raw: str) -> str:
    s = normalize_text(raw) if CFG["use_normalize"] else raw
    if CFG["auto_restore"] and approx_diacritic_ratio(s) < CFG["restore_threshold"]:
        s = restore_diacritics(s)
    if CFG["use_prefix"]:
        s = sentiment_prefix(s, max_tag=1)
    s = maybe_segment(s, use_seg=CFG["use_seg"])
    return s


@torch.inference_mode()
def infer(text: str) -> str:
    """
    Trả về chuỗi nhiều dòng (giống test_phobert.py):
      Text: ...
      Pred: ...
      Conf: ...
      Probs:
        very_negative : ...
        ...
    """
    _ensure()
    assert _tok is not None and _mdl is not None

    processed = _preprocess(text)
    enc = _tok([processed], truncation=True, padding=True,
               max_length=int(CFG["max_len"]), return_tensors="pt")
    # Tất cả tensor ở CPU
    enc = {k: v.to(_DEVICE) for k, v in enc.items()}

    logits = _mdl(**enc).logits.detach().cpu().numpy()[0]

    # Optional neutral tweak (index 2 là 'neutral')
    if CFG["neutral_penalty"] != 0.0:
        logits[2] += float(CFG["neutral_penalty"])

    probs = _softmax(logits)
    pred_id = int(probs.argmax())
    pred_lbl = IDX2LABEL.get(pred_id, str(pred_id))
    conf = float(probs[pred_id])

    lines = [
        f"Text: {text}",
        f"Pred: {pred_lbl}",
        f"Conf: {conf:.6f}",
        "Probs:",
    ]
    for i, name in enumerate(LABELS_5):
        lines.append(f"  {name:15s}: {float(probs[i]):.6f}")
    return "\n".join(lines)