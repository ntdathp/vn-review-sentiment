#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math, random, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from inspect import signature

from datasets import Dataset, Features, Value, ClassLabel
import evaluate
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
try:
    from transformers import EarlyStoppingCallback
    HAS_EARLYSTOP = True
except Exception:
    HAS_EARLYSTOP = False

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

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

# Slang/abbr rules
_SL_RE = [
    (r"\bsp\b", "sản phẩm"),
    (r"\bte\b", "tệ"),
    (r"\bv(cl|kl|l|ch)\b", "rất"),
    (r"\b(k|ko|k0|kh)\b", "không"),
    (r"\bgiao\s*hn\b", "giao hàng"),
    (r"\bok(e+)?\b", "ok"),
    (r"\b10/10\b", "rất tốt"),
]
_RE_LIST = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _SL_RE]

REPL_DICT = {
    "đỉnh của chóp": "rất tốt",
    "mãi đỉnh": "rất tốt",
    "best of best": "rất tốt",
    "best choice": "rất tốt",
    "siêu siêu": "rất",
    "siêu thất vọng": "rất thất vọng",
    "ưng": "rất thích",
}

def normalize_text_vn(s: str) -> str:
    s = str(s).strip()
    for e in EMO_POS: s = s.replace(e, " EMO_POS ")
    for e in EMO_NEG: s = s.replace(e, " EMO_NEG ")
    for rx, rep in _RE_LIST: s = rx.sub(rep, s)
    for k, v in REPL_DICT.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)
    return s

def count_lexicon(s: str):
    txt = s.lower()
    pos = sum(1 for p in POS_PATTERNS if re.search(p, txt))
    neg = sum(1 for p in NEG_PATTERNS if re.search(p, txt))
    pos += txt.count("emo_pos"); neg += txt.count("emo_neg")
    return pos, neg

def sentiment_prefix_train_only(s: str, max_tag=1):
    pos, neg = count_lexicon(s)
    pos = min(pos, max_tag); neg = min(neg, max_tag)
    prefix = []
    if pos>0: prefix.append(f"__POS{pos}__")
    if neg>0: prefix.append(f"__NEG{neg}__")
    return (" ".join(prefix) + " " + s) if prefix else s

# ===== Data load / split =====
def dedupe_dataset(ds: Dataset):
    df = ds.to_pandas()
    b = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    a = len(df)
    if a < b: print(f"[Dedup] removed {b-a} duplicates")
    feats = Features({"text": Value("string"), "labels": ClassLabel(names=LABELS_5)})
    return Dataset.from_pandas(df[["text","labels"]], preserve_index=False).cast(feats)

def load_train_val(csv_path, use_normalize=True, use_prefix_train=True, stratify=True,
                   augment=False, aug_prob=0.25):
    df = pd.read_csv(csv_path)
    assert {"text","label"}.issubset(df.columns), "CSV cần cột text,label"
    df = df.dropna(subset=["text","label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS_5)].copy()
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

    if use_normalize:
        df_tr["text"] = df_tr["text"].apply(normalize_text_vn)
        df_te["text"] = df_te["text"].apply(normalize_text_vn)

    if use_prefix_train:
        df_tr["text"] = df_tr["text"].apply(lambda s: sentiment_prefix_train_only(s, max_tag=1))

    if augment:
        df_tr = apply_augmentation(df_tr, p=aug_prob)

    feats = Features({"text": Value("string"), "labels": ClassLabel(names=LABELS_5)})
    ds_tr = Dataset.from_pandas(df_tr[["text","labels"]], preserve_index=False).cast(feats)
    ds_te = Dataset.from_pandas(df_te[["text","labels"]], preserve_index=False).cast(feats)

    ds_tr = dedupe_dataset(ds_tr)
    ds_te = dedupe_dataset(ds_te)

    print("Train size:", len(ds_tr), "| Val size:", len(ds_te))
    return ds_tr, ds_te

def load_eval_dataset(csv_path, use_normalize=True, apply_prefix=False):
    df = pd.read_csv(csv_path)
    assert {"text","label"}.issubset(df.columns)
    df = df.dropna(subset=["text","label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS_5)].copy()
    if use_normalize: df["text"] = df["text"].apply(normalize_text_vn)
    if apply_prefix:  df["text"] = df["text"].apply(lambda s: sentiment_prefix_train_only(s, max_tag=1))
    df["labels"] = df["label"].map(LBL2ID).astype(int)
    feats = Features({"text": Value("string"), "labels": ClassLabel(names=LABELS_5)})
    return Dataset.from_pandas(df[["text","labels"]], preserve_index=False).cast(feats), df

# ===== Augmentation =====
def augment_text_slang(s: str) -> str:
    ops = []
    if random.random() < 0.5: ops.append(lambda x: re.sub(r"\btệ\b", "tệ vcl", x, count=1))
    if random.random() < 0.3: ops.append(lambda x: re.sub(r"\bsản phẩm\b", "sp", x, count=1))
    if random.random() < 0.3: ops.append(lambda x: x + " 😡")
    if random.random() < 0.2: ops.append(lambda x: re.sub(r"\bok\b", "okeee", x, count=1, flags=re.IGNORECASE))
    for f in ops: s = f(s)
    return s

def apply_augmentation(df_tr: pd.DataFrame, p=0.25) -> pd.DataFrame:
    rows = [ {"text": r["text"], "labels": r["labels"]} for _, r in df_tr.iterrows() ]
    for _, r in df_tr.iterrows():
        if random.random() <= p:
            rows.append({"text": augment_text_slang(r["text"]), "labels": r["labels"]})
    return pd.DataFrame(rows)

# ===== Consistency support =====
def build_pairs_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_in.iterrows():
        raw = str(r["text"]).strip()
        nor = normalize_text_vn(raw)
        rows.append({"text_raw": raw, "text_nor": nor, "labels": r["labels"]})
    return pd.DataFrame(rows)

class PairDataCollator:
    """Robust stacker: handles tensors, lists, and numpy arrays without tokenizer.pad."""
    def __call__(self, features):
        import torch, numpy as np
        out = {}
        keys = features[0].keys()
        for k in keys:
            vals = [f[k] for f in features]
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.stack([v.long() for v in vals], dim=0)
            elif isinstance(v0, (list, tuple, np.ndarray)):
                out[k] = torch.as_tensor(vals, dtype=torch.long)
            else:
                # scalar labels etc.
                out[k] = torch.as_tensor(vals, dtype=torch.long)
        return out

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

class ConsistencyTrainer(Trainer):
    def __init__(self, kl_weight=0.5, **kw):
        super().__init__(**kw)
        self.kl_weight = kl_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=getattr(self.args, "label_smoothing_factor", 0.0))
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        y = inputs["labels"]
        out_raw = model(input_ids=inputs["raw_input_ids"], attention_mask=inputs["raw_attention_mask"])
        out_nor = model(input_ids=inputs["nor_input_ids"], attention_mask=inputs["nor_attention_mask"])
        logits_raw, logits_nor = out_raw.logits, out_nor.logits
        ce_raw = self.ce(logits_raw, y)
        ce_nor = self.ce(logits_nor, y)
        p = F.log_softmax(logits_raw, dim=-1); q = F.log_softmax(logits_nor, dim=-1)
        p_exp, q_exp = p.exp(), q.exp()
        kl_pq = F.kl_div(p, q_exp, reduction="batchmean")
        kl_qp = F.kl_div(q, p_exp, reduction="batchmean")
        kl = 0.5 * (kl_pq + kl_qp)
        loss = ce_raw + ce_nor + self.kl_weight * kl
        return (loss, {"logits": logits_raw}) if return_outputs else loss


def compute_class_weight_from_train(ds_train):
    import numpy as _np
    import torch as _torch
    y = _np.array(ds_train["labels"])
    counts = _np.bincount(y, minlength=len(LABELS_5)).astype(float)
    inv = 1.0 / _np.maximum(counts, 1.0)
    w = inv / inv.sum() * len(LABELS_5)
    print("[ClassWeight] counts:", counts.tolist(), "-> weight:", w.tolist())
    return _torch.tensor(w, dtype=_torch.float)
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

def can_use_early_stopping():
    try:
        sig = signature(TrainingArguments.__init__).parameters
    except Exception:
        return False
    need = ["evaluation_strategy", "save_strategy", "metric_for_best_model", "load_best_model_at_end"]
    return all(k in sig for k in need)

def build_training_args(args, len_train):
    sig = signature(TrainingArguments.__init__).parameters
    def supports(k): return k in sig
    kw = {}
    base = {"output_dir": args.output_dir, "learning_rate": args.lr, "num_train_epochs": args.epochs, "weight_decay": args.weight_decay}
    for k, v in base.items():
        if supports(k): kw[k] = v
    if supports("per_device_train_batch_size"):
        kw["per_device_train_batch_size"] = args.batch_size
        if supports("per_device_eval_batch_size"):
            kw["per_device_eval_batch_size"] = args.batch_size
    else:
        if supports("per_gpu_train_batch_size"): kw["per_gpu_train_batch_size"] = args.batch_size
        if supports("per_gpu_eval_batch_size"): kw["per_gpu_eval_batch_size"] = args.batch_size
    if supports("fp16"): kw["fp16"] = torch.cuda.is_available()
    if supports("lr_scheduler_type"): kw["lr_scheduler_type"] = "cosine"
    if supports("warmup_ratio"): kw["warmup_ratio"] = 0.06
    if supports("seed"): kw["seed"] = 42
    if supports("logging_steps"): kw["logging_steps"] = 50
    if supports("logging_first_step"): kw["logging_first_step"] = True
    if supports("max_grad_norm"): kw["max_grad_norm"] = 1.0
    if supports("report_to"): kw["report_to"] = []
    if can_use_early_stopping():
        kw["evaluation_strategy"] = "epoch"; kw["save_strategy"] = "epoch"
        if supports("load_best_model_at_end"): kw["load_best_model_at_end"] = True
        if supports("metric_for_best_model"): kw["metric_for_best_model"] = "eval_macro_f1"
        if supports("greater_is_better"): kw["greater_is_better"] = True
        if supports("save_total_limit"): kw["save_total_limit"] = 2
    else:
        steps_per_epoch = max(1, math.ceil(len_train / max(1, args.batch_size)))
        if supports("evaluate_during_training"): kw["evaluate_during_training"] = True
        if supports("eval_steps"): kw["eval_steps"] = steps_per_epoch
        if supports("save_steps"): kw["save_steps"] = steps_per_epoch
        if supports("logging_steps"): kw["logging_steps"] = max(1, steps_per_epoch // 2)
        if supports("load_best_model_at_end"): kw["load_best_model_at_end"] = False
        if supports("save_total_limit"): kw["save_total_limit"] = 2
    if getattr(args, 'consistency', False) and supports('remove_unused_columns'):
        kw['remove_unused_columns'] = False
    return TrainingArguments(**kw)

# ===== Main =====
def main(args):
    set_seed(42)

    ds_train, ds_val = load_train_val(
        args.train_csv,
        use_normalize=args.normalize,
        use_prefix_train=args.prefix,
        stratify=(not args.no_stratify),
        augment=args.augment,
        aug_prob=args.aug_prob
    )

    model_name = "vinai/phobert-base" if args.model_name is None else args.model_name
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    added_tokens = []
    for t in ["EMO_POS", "EMO_NEG"] + (["__POS1__", "__NEG1__"] if args.prefix else []):
        if tok.convert_tokens_to_ids(t) == tok.unk_token_id:
            added_tokens.append(t)
    if added_tokens:
        tok.add_tokens(added_tokens, special_tokens=False)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABELS_5),
        id2label=ID2LBL, label2id=LBL2ID,
        hidden_dropout_prob=args.hidden_dropout,
        attention_probs_dropout_prob=args.attn_dropout
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    if added_tokens:
        model.resize_token_embeddings(len(tok))

    trainer_sig = signature(Trainer.__init__).parameters
    use_processing_class = ("processing_class" in trainer_sig)

    if not args.consistency:
        def tok_fn(batch):
            return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)
        ds_train_tok = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
        ds_val_tok   = ds_val.map(tok_fn,   batched=True, remove_columns=["text"])
        ds_train_tok.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        ds_val_tok.set_format(type="torch",   columns=["input_ids","attention_mask","labels"])
        data_collator = DataCollatorWithPadding(tokenizer=tok, padding="max_length", max_length=args.max_len)
    else:
        df = pd.read_csv(args.train_csv)
        df = df.dropna(subset=["text","label"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["label"].isin(LABELS_5)].copy()
        df["labels"] = df["label"].map(LBL2ID).astype(int)
        if "group" in df.columns:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            idx_tr, _ = next(gss.split(df, groups=df["group"]))
            df_tr = df.iloc[idx_tr].copy()
        else:
            df_tr, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df["labels"])
        if args.prefix:
            df_tr["text"] = df_tr["text"].apply(lambda s: sentiment_prefix_train_only(s, max_tag=1))
        df_pairs = build_pairs_dataframe(df_tr)
        feats = Features({"text_raw": Value("string"), "text_nor": Value("string"), "labels": ClassLabel(names=LABELS_5)})
        ds_train_pairs = Dataset.from_pandas(df_pairs, preserve_index=False).cast(feats)

        def tok_pair(batch):
            A = tok(batch["text_raw"], truncation=True, padding="max_length", max_length=args.max_len)
            B = tok(batch["text_nor"], truncation=True, padding="max_length", max_length=args.max_len)
            return {
                "raw_input_ids": A["input_ids"], "raw_attention_mask": A["attention_mask"],
                "nor_input_ids": B["input_ids"], "nor_attention_mask": B["attention_mask"],
                "labels": batch["labels"]
            }
        ds_train_tok = ds_train_pairs.map(tok_pair, batched=True, remove_columns=["text_raw","text_nor"])
        ds_train_tok.set_format(type="torch", columns=["raw_input_ids","raw_attention_mask","nor_input_ids","nor_attention_mask","labels"])

        def tok_fn(batch):
            return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)
        ds_val_tok = ds_val.map(tok_fn, batched=True, remove_columns=["text"])
        ds_val_tok.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

        data_collator = PairDataCollator()

    training_args = build_training_args(args, len_train=len(ds_train_tok))

    base_for_weight = ds_train  # simpler & robust; label distribution is equivalent for weighting
    try:
        class_weight = compute_class_weight_from_train(base_for_weight) if args.class_weight else None
    except Exception as e:
        print('[ClassWeight] fallback due to:', e)
        class_weight = None
    if class_weight is not None and torch.cuda.is_available():
        class_weight = class_weight.cuda()

    callbacks_list = []
    if HAS_EARLYSTOP and can_use_early_stopping():
        callbacks_list = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok,
        compute_metrics=build_metrics(),
        callbacks=callbacks_list if callbacks_list else None
    )
    trainer_kwargs = {k:v for k,v in trainer_kwargs.items() if v is not None}
    if use_processing_class:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    if args.consistency:
        trainer = ConsistencyTrainer(kl_weight=args.kl_weight, **trainer_kwargs)
    else:
        trainer = CustomTrainer(
            use_focal=args.focal,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            class_weight=class_weight,
            **trainer_kwargs
        )

    trainer.train()

    try:
        eval_out = trainer.evaluate()
        print("\n[Evaluate on VAL] ->", eval_out)
    except Exception as e:
        print("[Evaluate] skipped:", e)

    val_preds = trainer.predict(ds_val_tok)
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

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.challenge_csv:
        ds_chal, df_chal_raw = load_eval_dataset(
            args.challenge_csv, use_normalize=args.normalize, apply_prefix=False
        )
        def tok_fn(batch):
            return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)
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
    p.add_argument("--train_csv", default="/home/dat/llm_ws/data/train/vn_reviews_train_clean_5cls_diverse_v3.csv")
    p.add_argument("--challenge_csv", default="/home/dat/llm_ws/data/test/vn_product_reviews_test_100_challenge.csv")
    p.add_argument("--output_dir", default="phobert_5cls_clean")
    p.add_argument("--model_name", default=None)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--hidden_dropout", type=float, default=0.2)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--class_weight", action="store_true", default=True)
    p.add_argument("--focal", action="store_true", default=True)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--prefix", action="store_true", default=True)
    p.add_argument("--augment", action="store_true", default=False)
    p.add_argument("--aug_prob", type=float, default=0.25)
    p.add_argument("--consistency", action="store_true", default=False)
    p.add_argument("--kl_weight", type=float, default=0.5)
    p.add_argument("--no_stratify", action="store_true")
    args = p.parse_args()
    main(args)
