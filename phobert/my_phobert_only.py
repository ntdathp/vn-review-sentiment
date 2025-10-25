# my_phobert_only.py
# -*- coding: utf-8 -*-
"""
PhoBERT wrapper for Chat Toolbox GUI, aligned with test_phobert.py preprocessing:
normalize -> (auto-restore diacritics) -> sentiment_prefix -> maybe_segment -> encode -> predict.

Exposes:
  - infer(text: str) -> str     # returns multiline string similar to test_phobert.py
  - set_config(**kwargs)        # optional runtime tweaks (max_len, use_seg, etc.)
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- Text pipeline (reuse exactly like test_phobert.py) ----
from textproc import (
    LABELS_5, normalize_text, maybe_segment, sentiment_prefix,
    approx_diacritic_ratio, restore_diacritics
)

# ===== Model dir autodetect (handles both of your paths) =====
_DEF_CANDIDATES: List[str] = [
    os.environ.get("PHOBERT_MODEL_DIR", ""),                 # allow env override
    "/home/dat/llm_ws/phobert/phobert_5cls_clean",           # test_phobert.py default
]
MODEL_DIR_PHOBERT = next((p for p in _DEF_CANDIDATES if p and os.path.isdir(p)), "")

# ===== Labels (reuse from textproc) =====
IDX2LABEL: Dict[int, str] = {i: l for i, l in enumerate(LABELS_5)}

# ===== Runtime config (mirrors test_phobert.py flags) =====
CFG = {
    "max_len": 160,           # --max_len
    "use_seg": False,         # --use_seg
    "use_normalize": True,    # --normalize (default True in test script)
    "use_prefix": True,       # NOT --no_prefix -> default True
    "auto_restore": True,     # NOT --no_auto_restore -> default True
    "restore_threshold": 0.03,# --restore_threshold
    "neutral_penalty": 0.0,   # --neutral_penalty
}

_device: Optional[torch.device] = None
_tok: Optional[AutoTokenizer] = None
_mdl: Optional[AutoModelForSequenceClassification] = None

def set_config(**kwargs: Any):
    """Update runtime config, e.g., set_config(use_seg=True, neutral_penalty=-0.05)."""
    for k, v in kwargs.items():
        if k in CFG:
            CFG[k] = v

def _device_auto() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def _ensure():
    """Load tokenizer & model once. Use HF local-from-pretrained correctly."""
    global _tok, _mdl, MODEL_DIR_PHOBERT
    if _tok is not None and _mdl is not None:
        return
    if not MODEL_DIR_PHOBERT:
        # Last resort: try any candidate that exists now (maybe created after import time)
        for p in _DEF_CANDIDATES:
            if p and os.path.isdir(p):
                MODEL_DIR_PHOBERT = p
                break
    if not MODEL_DIR_PHOBERT or not os.path.isdir(MODEL_DIR_PHOBERT):
        raise FileNotFoundError(
            "Checkpoint not found. Set PHOBERT_MODEL_DIR or create one of:\n  - /home/dat/llm_ws/phobert/phobert_5cls_clean\n  - /home/dat/llm_ws/phobert_5cls_clean"
        )
    # test_phobert.py uses use_fast=False; keep same for parity
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR_PHOBERT, use_fast=False)
    _mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_PHOBERT)
    _mdl.to(_device_auto()).eval()

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, keepdims=True)

def _preprocess(raw: str) -> str:
    s = normalize_text(raw) if CFG["use_normalize"] else raw
    if CFG["auto_restore"] and approx_diacritic_ratio(s) < CFG["restore_threshold"]:
        s = restore_diacritics(s)
    if CFG["use_prefix"]:
        s = sentiment_prefix(s, max_tag=1)  # y hệt test_phobert
    s = maybe_segment(s, use_seg=CFG["use_seg"])
    return s

@torch.inference_mode()
def infer(text: str) -> str:
    """
    Returns a multi-line string:
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
    enc = {k: v.to(_device_auto()) for k, v in enc.items()}
    logits = _mdl(**enc).logits.detach().cpu().numpy()[0]

    # Optional neutral tweak (index 2 is 'neutral' in LABELS_5)
    if CFG["neutral_penalty"] != 0.0:
        logits[2] += float(CFG["neutral_penalty"])

    probs = _softmax(logits)
    pred_id = int(probs.argmax())
    pred_lbl = IDX2LABEL.get(pred_id, str(pred_id))
    conf = float(probs[pred_id])

    # Build the same style output as test_phobert.py
    lines = [
        f"Text: {text}",
        f"Pred: {pred_lbl}",
        f"Conf: {conf:.6f}",
        "Probs:",
    ]
    for i, name in enumerate(LABELS_5):
        lines.append(f"  {name:15s}: {float(probs[i]):.6f}")
    return "\n".join(lines)
