#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math, random, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from inspect import signature

from datasets import Dataset, Features, Value, ClassLabel
import evaluate
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
try:
    from transformers import EarlyStoppingCallback
    HAS_EARLYSTOP = True
except Exception:
    HAS_EARLYSTOP = False

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from textproc import (LABELS_5, normalize_text, sentiment_prefix, maybe_segment)


# ===== Labels (5-class) =====
LABELS_5 = ["very_negative","negative","neutral","positive","very_positive"]
LBL2ID = {l:i for i,l in enumerate(LABELS_5)}
ID2LBL = {i:l for l,i in LBL2ID.items()}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ===== Lexicon / normalize =====
EMO_POS = ["🤩","🥰","😍","❤️","👍","😎","👌","✨","🔥","💯"]
EMO_NEG = ["😱","😡","🤬","💩","👎","😤","😞","😭"]
POS_PATTERNS = [
    r"\bđỉnh(?:\s+của\s+chóp)?\b", r"\bmãi\s*đỉnh\b", r"\bcực\s*phẩm\b",
    r"\bbest\s+of\s+best\b", r"\bbest\s+choice\b", r"\bquá\s*ok(e+)\b",
    r"\bok\s*phết\b", r"\bổn\s*áp\b", r"\brecommend\s*mạnh\b",
    r"\byêu\s+lắm\s+lun\b", r"\b10/10\b", r"\bperfect\b",
    r"\bkhông\s+chê\s+vào\s+đâu\s+được?\b", r"\bquá\s+yêu\b", r"\btuyệt\s*vời\b"
]
NEG_PATTERNS = [
    r"\bhỏng\s+ngay\s+lần\s*1\b", r"\bthảm\s*họa\b", r"\bkinh\s*dị\b",
    r"\blừa\s*đảo\b", r"\bbực\s*mình\s*vl\b", r"\bbực\s*mình\b", r"\bức\s*xúc\b",
    r"\bvứt\s*sọt\s*rác\b", r"\btệ\s*hại\b", r"\bquá\s*tệ\b",
    r"\bgiao\s*hàng\s*(lâu|delay|kẹt\s*mãi)\b"
]

def normalize_text(s: str) -> str:
    s = str(s).strip()
    for e in EMO_POS: s = s.replace(e, " EMO_POS ")
    for e in EMO_NEG: s = s.replace(e, " EMO_NEG ")
    repl = {
        "vl": "rất", "okeee": "ok", "ưng": "rất thích",
        "siêu siêu": "rất", "siêu thất vọng": "rất thất vọng",
        "mãi đỉnh": "rất tốt", "best of best": "rất tốt", "best choice": "rất tốt",
        "đỉnh của chóp": "rất tốt",
    }
    for k,v in repl.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)
    return s

def count_lexicon(s: str):
    txt = s.lower()
    pos = sum(1 for p in POS_PATTERNS if re.search(p, txt))
    neg = sum(1 for p in NEG_PATTERNS if re.search(p, txt))
    pos += txt.count("emo_pos"); neg += txt.count("emo_neg")
    return pos, neg

def sentiment_prefix(s: str, max_tag=1):
    pos, neg = count_lexicon(s)
    pos = min(pos, max_tag); neg = min(neg, max_tag)
    prefix = []
    if pos>0: prefix.append(f"__POS{pos}__")
    if neg>0: prefix.append(f"__NEG{neg}__")
    return (" ".join(prefix) + " " + s) if prefix else s

def maybe_segment(text, use_seg=False):
    if not use_seg: return text
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")

# ===== Data load / split =====
def dedupe_dataset(ds: Dataset):
    df = ds.to_pandas()
    b = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    a = len(df)
    if a < b: print(f"[Dedup] removed {b-a} duplicates")
    feats = Features({"text": Value("string"), "labels": ClassLabel(names=LABELS_5)})
    return Dataset.from_pandas(df[["text","labels"]], preserve_index=False).cast(feats)

def load_train_val(csv_path, use_seg=False, use_normalize=True,
                   use_prefix_train=True, stratify=True):
    df = pd.read_csv(csv_path)
    assert {"text","label"}.issubset(df.columns), "CSV cần cột text,label"
    df = df.dropna(subset=["text","label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS_5)].copy()
    if use_normalize: df["text"] = df["text"].apply(normalize_text)
    df["labels"] = df["label"].map(LBL2ID).astype(int)

    if "group" in df.columns:
        print("[Split] GroupShuffleSplit by 'group'")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_tr, idx_te = next(gss.split(df, groups=df["group"]))
        df_tr, df_te = df.iloc[idx_tr].copy(), df.iloc[idx_te].copy()
    else:
        if stratify:
            df_tr, df_te = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["labels"]
            )
        else:
            df_tr, df_te = train_test_split(df, test_size=0.2, random_state=42)

    if use_prefix_train:
        df_tr["text"] = df_tr["text"].apply(lambda s: sentiment_prefix(s, max_tag=1))

    if use_seg:
        from underthesea import word_tokenize
        df_tr["text"] = df_tr["text"].apply(lambda s: word_tokenize(s, format="text"))
        df_te["text"] = df_te["text"].apply(lambda s: word_tokenize(s, format="text"))

    feats = Features({"text": Value("string"), "labels": ClassLabel(names=LABELS_5)})
    ds_tr = Dataset.from_pandas(df_tr[["text","labels"]], preserve_index=False).cast(feats)
    ds_te = Dataset.from_pandas(df_te[["text","labels"]], preserve_index=False).cast(feats)

    ds_tr = dedupe_dataset(ds_tr)
    ds_te = dedupe_dataset(ds_te)

    print("Train size:", len(ds_tr), "| Val size:", len(ds_te))
    return ds_tr, ds_te

def load_eval_dataset(csv_path, use_seg=False, use_normalize=True, apply_prefix=False):
    df = pd.read_csv(csv_path)
    assert {"text","label"}.issubset(df.columns)
    df = df.dropna(subset=["text","label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS_5)].copy()
    if use_normalize: df["text"] = df["text"].apply(normalize_text)
    if apply_prefix:  df["text"] = df["text"].apply(lambda s: sentiment_prefix(s, max_tag=1))
    if use_seg:
        from underthesea import word_tokenize
        df["text"] = df["text"].apply(lambda s: word_tokenize(s, format="text"))
    df["labels"] = df["label"].map(LBL2ID).astype(int)
    feats = Features({"text": Value("string"), "labels": ClassLabel(names=LABELS_5)})
    return Dataset.from_pandas(df[["text","labels"]], preserve_index=False).cast(feats), df

# ===== Loss & Trainer =====
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.weight = weight
        self.reduction = reduction; self.label_smoothing = label_smoothing
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.weight, reduction="none",
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce
        return fl.mean() if self.reduction=="mean" else (fl.sum() if self.reduction=="sum" else fl)

class CustomTrainer(Trainer):
    def __init__(self, use_focal=False, gamma=2.0, label_smoothing=0.05,
                 class_weight=None, **kwargs):
        super().__init__(**kwargs)
        if use_focal:
            self.criterion = FocalLoss(gamma=gamma, weight=class_weight,
                                       label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weight,
                                                 label_smoothing=label_smoothing)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits = outputs.get("logits")
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_class_weight_from_train(ds_train):
    y = np.array(ds_train["labels"])
    counts = np.bincount(y, minlength=len(LABELS_5)).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    w = inv / inv.sum() * len(LABELS_5)
    print("[ClassWeight] counts:", counts.tolist(), "-> weight:", w.tolist())
    return torch.tensor(w, dtype=torch.float)

# ===== Metrics =====
def build_metrics():
    acc = evaluate.load("accuracy"); f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    return compute_metrics

# ===== Capability checks =====
def can_use_early_stopping():
    try:
        sig = signature(TrainingArguments.__init__).parameters
    except Exception:
        return False
    need = ["evaluation_strategy", "save_strategy", "metric_for_best_model", "load_best_model_at_end"]
    return all(k in sig for k in need)

# ===== Robust TrainingArguments builder (compat) =====
def build_training_args(args, len_train):
    sig = signature(TrainingArguments.__init__).parameters
    def supports(k): return k in sig

    kw = {}

    # core
    base = {
        "output_dir": args.output_dir,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
    }
    for k, v in base.items():
        if supports(k): kw[k] = v

    # batch sizes (new vs very old)
    if supports("per_device_train_batch_size"):
        kw["per_device_train_batch_size"] = args.batch_size
        if supports("per_device_eval_batch_size"):
            kw["per_device_eval_batch_size"] = args.batch_size
    else:
        if supports("per_gpu_train_batch_size"):
            kw["per_gpu_train_batch_size"] = args.batch_size
        if supports("per_gpu_eval_batch_size"):
            kw["per_gpu_eval_batch_size"] = args.batch_size

    # typical extras if available
    if supports("fp16"): kw["fp16"] = torch.cuda.is_available()
    if supports("lr_scheduler_type"): kw["lr_scheduler_type"] = "cosine"
    if supports("warmup_ratio"): kw["warmup_ratio"] = 0.06
    if supports("seed"): kw["seed"] = 42
    if supports("logging_steps"): kw["logging_steps"] = 50
    if supports("logging_first_step"): kw["logging_first_step"] = True
    if supports("max_grad_norm"): kw["max_grad_norm"] = 1.0
    if supports("report_to"): kw["report_to"] = []

    # ---------- VERSION COMPAT SWITCH ----------
    if can_use_early_stopping():
        # Newer HF → dùng theo epoch + best model
        kw["evaluation_strategy"] = "epoch"
        kw["save_strategy"] = "epoch"
        if supports("load_best_model_at_end"): kw["load_best_model_at_end"] = True
        if supports("metric_for_best_model"):  kw["metric_for_best_model"]  = "eval_macro_f1"
        if supports("greater_is_better"):      kw["greater_is_better"]      = True
        if supports("save_total_limit"):       kw["save_total_limit"]       = 2
    else:
        # Older HF → KHÔNG bật load_best_model_at_end để tránh lỗi mismatch
        steps_per_epoch = max(1, math.ceil(len_train / max(1, args.batch_size)))
        if supports("evaluate_during_training"): kw["evaluate_during_training"] = True
        if supports("eval_steps"):  kw["eval_steps"]  = steps_per_epoch
        if supports("save_steps"):  kw["save_steps"]  = steps_per_epoch
        if supports("logging_steps"): kw["logging_steps"] = max(1, steps_per_epoch // 2)
        if supports("load_best_model_at_end"): kw["load_best_model_at_end"] = False
        if supports("save_total_limit"): kw["save_total_limit"] = 2
    # -------------------------------------------

    return TrainingArguments(**kw)

# ===== Main =====
def main(args):
    set_seed(42)

    # Data
    ds_train, ds_val = load_train_val(
        args.train_csv,
        use_seg=args.use_seg,
        use_normalize=args.normalize,
        use_prefix_train=args.prefix,
        stratify=(not args.no_stratify)
    )

    # Tokenizer & Model
    model_name = "vinai/phobert-base" if args.model_name is None else args.model_name
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    ds_val   = ds_val.map(tok_fn,   batched=True, remove_columns=["text"])
    ds_train.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    ds_val.set_format(type="torch",   columns=["input_ids","attention_mask","labels"])

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABELS_5),
        id2label=ID2LBL, label2id=LBL2ID,
        hidden_dropout_prob=args.hidden_dropout,
        attention_probs_dropout_prob=args.attn_dropout
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # TrainingArguments (compat)
    training_args = build_training_args(args, len_train=len(ds_train))

    # Trainer init (compat: processing_class vs tokenizer)
    trainer_sig = signature(Trainer.__init__).parameters
    use_processing_class = ("processing_class" in trainer_sig)

    class_weight = compute_class_weight_from_train(ds_train) if args.class_weight else None
    if class_weight is not None and torch.cuda.is_available():
        class_weight = class_weight.cuda()

    callbacks_list = []
    if HAS_EARLYSTOP and can_use_early_stopping():
        callbacks_list = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=build_metrics(),
        callbacks=callbacks_list if callbacks_list else None
    )
    # remove None
    trainer_kwargs = {k:v for k,v in trainer_kwargs.items() if v is not None}

    if use_processing_class:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = CustomTrainer(
        use_focal=args.focal,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        class_weight=class_weight,
        **trainer_kwargs
    )

    # Train
    trainer.train()

    # Evaluate on VAL
    try:
        eval_out = trainer.evaluate()
        print("\n[Evaluate on VAL] ->", eval_out)
    except Exception as e:
        print("[Evaluate] skipped:", e)

    val_preds = trainer.predict(ds_val)
    y_true = val_preds.label_ids
    y_pred = val_preds.predictions.argmax(-1)
    probs  = torch.softmax(torch.tensor(val_preds.predictions), dim=-1).numpy()
    pmax   = probs.max(axis=1)

    print("\n[VAL] Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\n[VAL] Classification report:")
    print(classification_report(y_true, y_pred, target_names=LABELS_5, digits=4))

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame({
        "_true": [ID2LBL[int(i)] for i in y_true],
        "_pred": [ID2LBL[int(i)] for i in y_pred],
        "_pmax": pmax
    }).to_csv(os.path.join(args.output_dir, "val_predictions.csv"), index=False)

    # Save model
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Challenge (optional)
    if args.challenge_csv:
        ds_chal, df_chal_raw = load_eval_dataset(
            args.challenge_csv, use_seg=args.use_seg,
            use_normalize=args.normalize, apply_prefix=False
        )
        ds_chal = ds_chal.map(tok_fn, batched=True, remove_columns=["text"])
        ds_chal.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

        chal_preds = trainer.predict(ds_chal)
        cy = chal_preds.label_ids
        cp = chal_preds.predictions.argmax(-1)
        cprobs = torch.softmax(torch.tensor(chal_preds.predictions), dim=-1).numpy()
        cpmax  = cprobs.max(axis=1)

        print("\n=== Challenge set ===")
        print(confusion_matrix(cy, cp))
        print(classification_report(cy, cp, target_names=LABELS_5, digits=4))

        df_out = df_chal_raw.copy()
        df_out["_true"] = [ID2LBL[int(i)] for i in cy]
        df_out["_pred"] = [ID2LBL[int(i)] for i in cp]
        df_out["_pmax"] = cpmax
        out_csv = os.path.join(args.output_dir, "challenge_predictions.csv")
        df_out.to_csv(out_csv, index=False)
        print("[Saved] Challenge predictions ->", out_csv)

        df_err = df_out[df_out["_true"] != df_out["_pred"]]
        df_err.to_csv(os.path.join(args.output_dir, "challenge_misclassified.csv"), index=False)
        print("[Saved] Misclassified ->", os.path.join(args.output_dir, "challenge_misclassified.csv"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="/home/dat/llm_ws/data/train/train.csv")
    p.add_argument("--challenge_csv", default="/home/dat/llm_ws/data/test/test.csv")
    p.add_argument("--output_dir", default="phobert_5cls_clean")
    p.add_argument("--model_name", default=None, help="vinai/phobert-base hoặc phobert-large")
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--hidden_dropout", type=float, default=0.2)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--class_weight", action="store_true", default=True)
    p.add_argument("--focal", action="store_true", default=True)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--prefix", action="store_true", default=True, help="Chỉ áp dụng cho TRAIN")
    p.add_argument("--use_seg", action="store_true")
    p.add_argument("--no_stratify", action="store_true")
    args = p.parse_args()
    main(args)
