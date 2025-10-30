#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from textproc import (
    LABELS_5, normalize_text, maybe_segment, sentiment_prefix,
    approx_diacritic_ratio, restore_diacritics
)

ID2LBL = {i: l for i, l in enumerate(LABELS_5)}

def softmax(x):
    x = x - np.max(x, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, keepdims=True)

def main():
    ap = argparse.ArgumentParser(description="Quick inference PhoBERT 5 lớp (CPU/GPU).")
    ap.add_argument("--model_dir", default="/home/dat/llm_ws/phobert/phobert_5cls_clean")
    ap.add_argument("--text", required=True)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--use_seg", action="store_true")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no_auto_restore", action="store_true")
    ap.add_argument("--restore_threshold", type=float, default=0.03)
    ap.add_argument("--neutral_penalty", type=float, default=0.0)
    ap.add_argument("--no_prefix", action="store_true",
                    help="Tắt sentiment prefix ở infer (mặc định bật để khớp train).")
    # NEW: ép device
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Chọn thiết bị chạy: cpu / cuda / auto.")
    # (tuỳ chọn) giới hạn thread CPU cho ổn định
    ap.add_argument("--cpu_threads", type=int, default=0,
                    help="Số thread CPU (0 = mặc định PyTorch).")
    args = ap.parse_args()

    # Chọn device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Yêu cầu --device=cuda nhưng CUDA không khả dụng.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    # Luôn nạp model ở dtype float32; nếu CPU thì map_location='cpu'
    load_kwargs = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
    if device.type == "cpu":
        load_kwargs["device_map"] = {"": "cpu"}  # tránh tự map sang CUDA
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, **load_kwargs
    ).to(device).eval()

    raw = args.text
    # 1) normalize
    s = normalize_text(raw) if args.normalize else raw
    # 2) auto-restore (nếu cần)
    if not args.no_auto_restore and approx_diacritic_ratio(s) < args.restore_threshold:
        restored = restore_diacritics(s)
        print(f'[INFO] Auto-restore diacritics → "{restored}"')
        s = restored
    # 3) prefix (đồng bộ với train)
    if not args.no_prefix:
        s = sentiment_prefix(s, max_tag=1)
    # 4) segmentation (nếu dùng)
    s = maybe_segment(s, use_seg=args.use_seg)

    with torch.inference_mode():
        enc = tok([s], truncation=True, padding=True,
                  max_length=args.max_len, return_tensors="pt")
        # chuyển tensor sang đúng device
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits
        # đưa về CPU để xử lý numpy
        logits = logits.detach().cpu().numpy()[0]

    # (tùy chọn) điều chỉnh neutral
    if args.neutral_penalty != 0.0:
        logits[2] += args.neutral_penalty  # 'neutral' index 2

    probs = softmax(logits)
    pred_id = int(probs.argmax())
    pred_lbl = ID2LBL[pred_id]

    print(f"[INFO] Device: {device}")
    print("Text:", raw)
    print("Pred:", pred_lbl)
    print("Conf:", f"{float(probs[pred_id]):.6f}")
    print("Probs:")
    for i, name in enumerate(LABELS_5):
        print(f"  {name:15s}: {float(probs[i]):.6f}")

if __name__ == "__main__":
    main()
