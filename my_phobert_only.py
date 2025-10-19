# my_phobert_only.py
# -*- coding: utf-8 -*-
"""
Minimal wrapper exposing infer(text) for the Chat Toolbox GUI.
Edit MODEL_DIR_PHOBERT and IDX2LABEL to match your checkpoint.
"""

from __future__ import annotations
from typing import Optional, Dict
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====== EDIT ME ======
MODEL_DIR_PHOBERT = "/home/dat/llm_ws/phobert_5cls_clean"

IDX2LABEL: Dict[int, str] = {
    0: "very_negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very_positive",
}

_device: Optional[torch.device] = None
_tok: Optional[AutoTokenizer] = None
_mdl: Optional[AutoModelForSequenceClassification] = None

def _device_auto() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def _ensure():
    global _tok, _mdl
    if _tok is None or _mdl is None:
        if not os.path.isdir(MODEL_DIR_PHOBERT):
            raise FileNotFoundError(f"Checkpoint not found: {MODEL_DIR_PHOBERT}")
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR_PHOBERT, use_fast=True)
        _mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_PHOBERT)
        _mdl.to(_device_auto()).eval()

@torch.inference_mode()
def infer(text: str) -> str:
    _ensure()
    assert _tok and _mdl
    enc = _tok(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(_device_auto()) for k, v in enc.items()}
    logits = _mdl(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    conf, idx = torch.max(probs, dim=-1)
    label = IDX2LABEL.get(idx.item(), str(idx.item()))
    return f"{label} (p={conf.item():.3f})"
