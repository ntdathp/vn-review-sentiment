#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math, random, argparse, csv
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from inspect import signature

from datasets import Dataset, Features, Value, ClassLabel
import evaluate
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
)
try:
    from transformers import EarlyStoppingCallback
    HAS_EARLYSTOP = True
except Exception:
    HAS_EARLYSTOP = False

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# ===== Try importing from textproc, but fall back to local defs =====
try:
    from textproc import (LABELS_5 as _LBL5, normalize_text as _norm_txt,
                          sentiment_prefix as _sent_prefix, maybe_segment as _maybe_seg)
    _HAS_TEXTPROC = True
except Exception:
    _HAS_TEXTPROC = False
    _LBL5 = None
    _norm_txt = None
    _sent_prefix = None
    _maybe_seg = None

# ===== Labels (5-class) =====
LABELS_5 = _LBL5 if _HAS_TEXTPROC and _LBL5 is not None else \
    ["very_negative","negative","neutral","positive","very_positive"]
LBL2ID = {l:i for i,l in enumerate(LABELS_5)}
ID2LBL = {i:l for l,i in LBL2ID.items()}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ===== Lexicon / normalize (fallback) =====
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
    if _HAS_TEXTPROC and _norm_txt is not None:
        return _norm_txt(s)
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
    if _HAS_TEXTPROC and _sent_prefix is not None:
        return _sent_prefix(s, max_tag=max_tag)
    pos, neg = count_lexicon(s)
    pos = min(pos, max_tag); neg = min(neg, max_tag)
    prefix = []
    if pos>0: prefix.append(f"__POS{pos}__")
    if neg>0: prefix.append(f"__NEG{neg}__")
    return (" ".join(prefix) + " " + s) if prefix else s

def maybe_segment(text, use_seg=False):
    if not use_seg: return text
    if _HAS_TEXTPROC and _maybe_seg is not None:
        return _maybe_seg(text, use_seg=True)
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

def load_train_val(csv_path, use_seg=False, use_normalize=False,
                   use_prefix_train=False, stratify=True,
                   val_ratio=0.2, seed=42):
    df = pd.read_csv(csv_path)
    assert {"text","label"}.issubset(df.columns), "CSV cần cột text,label"
    df = df.dropna(subset=["text","label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS_5)].copy()
    if use_normalize: df["text"] = df["text"].apply(normalize_text)
    df["labels"] = df["label"].map(LBL2ID).astype(int)

    if "group" in df.columns:
        print("[Split] GroupShuffleSplit by 'group']")
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        idx_tr, idx_te = next(gss.split(df, groups=df["group"]))
        df_tr, df_te = df.iloc[idx_tr].copy(), df.iloc[idx_te].copy()
    else:
        if stratify:
            df_tr, df_te = train_test_split(
                df, test_size=val_ratio, random_state=seed, stratify=df["labels"]
            )
        else:
            df_tr, df_te = train_test_split(df, test_size=val_ratio, random_state=seed)

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

    print(f"Train size: {len(ds_tr)} | Val size: {len(ds_te)} (val_ratio={val_ratio})")
    return ds_tr, ds_te

def load_eval_or_test_dataframe(csv_path, use_seg=False, use_normalize=False, apply_prefix=False):
    df = pd.read_csv(csv_path)
    assert "text" in df.columns, "CSV test cần có cột text"
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()

    has_label = "label" in df.columns
    if has_label:
        df = df.dropna(subset=["label"]).copy()
        df = df[df["label"].isin(LABELS_5)].copy()

    if use_normalize: df["text"] = df["text"].apply(normalize_text)
    if apply_prefix:  df["text"] = df["text"].apply(lambda s: sentiment_prefix(s, max_tag=1))
    if use_seg:
        from underthesea import word_tokenize
        df["text"] = df["text"].apply(lambda s: word_tokenize(s, format="text"))

    if has_label:
        df["labels"] = df["label"].map(LBL2ID).astype(int)

    return df, has_label

def dataset_from_dataframe_for_eval(df, has_label: bool):
    if has_label:
        feats = Features({"text": Value("string"), "labels": ClassLabel(names=LABELS_5)})
        ds = Dataset.from_pandas(df[["text","labels"]], preserve_index=False).cast(feats)
    else:
        feats = Features({"text": Value("string")})
        ds = Dataset.from_pandas(df[["text"]], preserve_index=False).cast(feats)
    return ds

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
        labels = inputs.get("labels", None)
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits = outputs.get("logits")
        if labels is None:
            loss = logits.new_zeros(())
            return (loss, outputs) if return_outputs else loss
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_class_weight_from_train(ds_train):
    y = np.array(ds_train["labels"])
    counts = np.bincount(y, minlength=len(LABELS_5)).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    w = inv / inv.sum() * len(LABELS_5)
    print("[ClassWeight] counts:", counts.tolist(), "-> weight:", w.tolist())
    return torch.tensor(w, dtype=torch.float)

# ===== Metrics (with fallback) =====
def build_metrics():
    try:
        acc = evaluate.load("accuracy")
        f1  = evaluate.load("f1")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
                "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            }
        return compute_metrics
    except Exception as e:
        print("[evaluate] fallback to sklearn metrics due to:", e)
        from sklearn.metrics import f1_score, accuracy_score
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "macro_f1": float(f1_score(labels, preds, average="macro")),
            }
        return compute_metrics

# ===== Helper to write CSV rows =====
def _append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["global_step","epoch","phase","loss","loss_ema","accuracy","macro_f1","lr"])
        w.writerow(row)

# ===== Unified Callback: log train, and run VAL + TEST per epoch =====
class EvalAndLogCallback(TrainerCallback):
    def __init__(self, log_csv_path: str,
                 ds_test=None, test_has_label: bool=False,
                 id2lbl=None,
                 do_manual_eval_per_epoch: bool=False):
        super().__init__()
        self.log_csv_path = log_csv_path
        self.ds_test = ds_test
        self.test_has_label = test_has_label
        self.id2lbl = id2lbl or {}
        self.ema_beta = 0.98
        self.ema_loss = None
        self.last_logged_step = -1
        self.trainer = None
        self.do_manual_eval_per_epoch = do_manual_eval_per_epoch

    def attach_trainer(self, trainer):
        self.trainer = trainer

    # ------- Train logs -------
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        if state.global_step == self.last_logged_step:
            return
        self.last_logged_step = state.global_step

        loss = float(logs["loss"])
        self.ema_loss = loss if self.ema_loss is None else self.ema_beta*self.ema_loss + (1-self.ema_beta)*loss
        lr = logs.get("learning_rate", None)
        epoch = logs.get("epoch", state.epoch)
        print(f"[Train] epoch={epoch:.2f} step={state.global_step} loss={loss:.4f} ema={self.ema_loss:.4f} lr={lr if lr is not None else '-'}")
        _append_csv_row(self.log_csv_path, [
            int(state.global_step), float(epoch), "train",
            float(loss), float(self.ema_loss),
            "", "", float(lr) if lr is not None else ""
        ])

    # ------- Khi Trainer tự evaluate (bản mới) → log VAL và chạy TEST song song -------
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        ev_loss = metrics.get("eval_loss")
        ev_acc  = metrics.get("eval_accuracy")
        ev_f1   = metrics.get("eval_macro_f1")
        epoch = state.epoch
        print(f"[Val]   epoch={epoch:.2f} eval_loss={ev_loss:.4f} acc={ev_acc if ev_acc is not None else '-'} macro_f1={ev_f1 if ev_f1 is not None else '-'}")
        _append_csv_row(self.log_csv_path, [
            int(state.global_step), float(epoch), "val",
            float(ev_loss) if ev_loss is not None else "",
            "", float(ev_acc) if ev_acc is not None else "",
            float(ev_f1) if ev_f1 is not None else "", ""
        ])

        # Chạy TEST song song
        self._run_test_and_log(state)

    # ------- Với bản cũ không tự evaluate: tự gọi ở cuối mỗi epoch -------
    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.do_manual_eval_per_epoch:
            return
        if self.trainer is None:
            return
        # VAL
        metrics = self.trainer.evaluate()
        ev_loss = metrics.get("eval_loss")
        ev_acc  = metrics.get("eval_accuracy")
        ev_f1   = metrics.get("eval_macro_f1")
        epoch = state.epoch
        print(f"[Val*]  epoch={epoch:.2f} eval_loss={ev_loss:.4f} acc={ev_acc if ev_acc is not None else '-'} macro_f1={ev_f1 if ev_f1 is not None else '-'}")
        _append_csv_row(self.log_csv_path, [
            int(state.global_step), float(epoch), "val",
            float(ev_loss) if ev_loss is not None else "",
            "", float(ev_acc) if ev_acc is not None else "",
            float(ev_f1) if ev_f1 is not None else "", ""
        ])

        # TEST
        self._run_test_and_log(state)

    # ------- Helper: chạy test và log + lưu csv theo epoch -------
    def _run_test_and_log(self, state):
        if (self.ds_test is None) or (self.trainer is None):
            return
        epoch = state.epoch
        test_preds = self.trainer.predict(self.ds_test)
        pred_labels = test_preds.predictions.argmax(-1)
        logits = test_preds.predictions
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        pmax  = probs.max(axis=1)

        if self.test_has_label:
            cy = test_preds.label_ids
            cm = confusion_matrix(cy, pred_labels)
            print("\n=== TEST (labeled, epoch={}) ===".format(epoch))
            print(cm)
            print(classification_report(cy, pred_labels, target_names=list(self.id2lbl.values()) or LABELS_5, digits=4))
            # log summary (không tính macro_f1 ở đây để nhẹ nhàng — nếu cần, có thể tự tính)
            # Ở đây ghi phase=test, chỉ loss trống vì evaluate(test) không chạy loss.
            _append_csv_row(self.log_csv_path, [
                int(state.global_step), float(epoch), "test", "", "", "", "", ""
            ])
            # lưu csv theo epoch
            out_dir = self.trainer.args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"test_predictions_epoch{int(round(epoch))}.csv")
            df_out = pd.DataFrame({
                "_true": [self.id2lbl.get(int(i), str(int(i))) for i in cy],
                "_pred": [self.id2lbl.get(int(i), str(int(i))) for i in pred_labels],
                "_pmax": pmax
            })
            df_out.to_csv(out_csv, index=False)
            err_csv = os.path.join(out_dir, f"test_misclassified_epoch{int(round(epoch))}.csv")
            df_out[df_out["_true"] != df_out["_pred"]].to_csv(err_csv, index=False)
            print(f"[Saved] Test predictions -> {out_csv}")
            print(f"[Saved] Misclassified -> {err_csv}\n")
        else:
            print("\n=== TEST (no labels, epoch={}) ===".format(epoch))
            _append_csv_row(self.log_csv_path, [
                int(state.global_step), float(epoch), "test", "", "", "", "", ""
            ])
            out_dir = self.trainer.args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"test_predictions_epoch{int(round(epoch))}.csv")
            df_out = pd.DataFrame({
                "_pred": [self.id2lbl.get(int(i), str(int(i))) for i in pred_labels],
                "_pmax": pmax
            })
            df_out.to_csv(out_csv, index=False)
            print(f"[Saved] Test predictions (no label) -> {out_csv}\n")

# ===== Capability checks =====
def supports_args(*names):
    fields = getattr(TrainingArguments, "__dataclass_fields__", {}) or {}
    try:
        sig_params = signature(TrainingArguments.__init__).parameters
    except Exception:
        sig_params = {}
    return {n: (n in fields or n in sig_params) for n in names}

def can_use_early_stopping():
    sup = supports_args("evaluation_strategy","save_strategy",
                        "metric_for_best_model","load_best_model_at_end")
    return all(sup.values())

# ===== Robust TrainingArguments builder (epoch/steps/no) =====
def build_training_args(args, len_train):
    sup = supports_args(
        "output_dir","learning_rate","num_train_epochs","weight_decay",
        "per_device_train_batch_size","per_device_eval_batch_size",
        "per_gpu_train_batch_size","per_gpu_eval_batch_size",
        "fp16","lr_scheduler_type","warmup_ratio","seed",
        "logging_steps","logging_first_step","max_grad_norm","report_to",
        "evaluation_strategy","save_strategy","eval_steps","save_steps",
        "evaluate_during_training",  # legacy
        "load_best_model_at_end","metric_for_best_model","greater_is_better",
        "save_total_limit"
    )
    def S(k): return sup.get(k, False)

    kw = {}

    # --- core ---
    if S("output_dir"):       kw["output_dir"] = args.output_dir
    if S("learning_rate"):    kw["learning_rate"] = args.lr
    if S("num_train_epochs"): kw["num_train_epochs"] = args.epochs
    if S("weight_decay"):     kw["weight_decay"] = args.weight_decay

    # --- batch ---
    if S("per_device_train_batch_size"):
        kw["per_device_train_batch_size"] = args.batch_size
        if S("per_device_eval_batch_size"):
            kw["per_device_eval_batch_size"] = args.batch_size
    else:
        if S("per_gpu_train_batch_size"):
            kw["per_gpu_train_batch_size"] = args.batch_size
        if S("per_gpu_eval_batch_size"):
            kw["per_gpu_eval_batch_size"] = args.batch_size

    # --- extras ---
    if S("fp16"):               kw["fp16"] = torch.cuda.is_available()
    if S("lr_scheduler_type"):  kw["lr_scheduler_type"] = "cosine"
    if S("warmup_ratio"):       kw["warmup_ratio"] = 0.06
    if S("seed"):               kw["seed"] = args.seed
    if S("logging_first_step"): kw["logging_first_step"] = True
    if S("max_grad_norm"):      kw["max_grad_norm"] = 1.0
    if S("report_to"):          kw["report_to"] = []

    strat = getattr(args, "eval_strategy", "epoch")
    have_evalsave = S("evaluation_strategy") and S("save_strategy")

    steps_per_epoch = max(1, math.ceil(len_train / max(1, args.batch_size)))
    if S("logging_steps"):
        kw["logging_steps"] = max(1, steps_per_epoch // 2)

    if have_evalsave:
        kw["evaluation_strategy"] = strat
        kw["save_strategy"] = strat
        if strat == "steps":
            step_int = max(1, int(getattr(args, "eval_steps", 200)))
            if S("eval_steps"): kw["eval_steps"] = step_int
            if S("save_steps"): kw["save_steps"] = step_int
            if S("logging_steps"): kw["logging_steps"] = max(1, step_int // 2)
        if strat != "no":
            if S("load_best_model_at_end"): kw["load_best_model_at_end"] = True
            if S("metric_for_best_model"):  kw["metric_for_best_model"]  = "eval_macro_f1"
            if S("greater_is_better"):      kw["greater_is_better"]      = True
            if S("save_total_limit"):       kw["save_total_limit"]       = 2
        else:
            if S("load_best_model_at_end"): kw["load_best_model_at_end"] = False
            if S("save_total_limit"):       kw["save_total_limit"]       = 2
    else:
        # Legacy: tự emulate theo steps_per_epoch
        if strat == "no":
            if S("evaluate_during_training"): kw["evaluate_during_training"] = False
            if S("load_best_model_at_end"):   kw["load_best_model_at_end"] = False
        else:
            if S("evaluate_during_training"): kw["evaluate_during_training"] = True
            if S("eval_steps"):               kw["eval_steps"]  = steps_per_epoch
            if S("save_steps"):               kw["save_steps"]  = steps_per_epoch
            if S("load_best_model_at_end"):   kw["load_best_model_at_end"] = False
        if S("save_total_limit"): kw["save_total_limit"] = 2

    return TrainingArguments(**kw), have_evalsave

# ===== Main =====
def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    # Data
    ds_train, ds_val = load_train_val(
        args.train_csv,
        use_seg=args.use_seg,
        use_normalize=args.normalize,
        use_prefix_train=args.prefix,
        stratify=(not args.no_stratify),
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Prepare TEST before training (để callback dùng giữa training)
    df_test, test_has_label = (None, False)
    ds_test = None
    if args.test_csv:
        df_test, test_has_label = load_eval_or_test_dataframe(
            args.test_csv, use_seg=args.use_seg,
            use_normalize=args.normalize, apply_prefix=False
        )
        # tokenization của test sẽ làm sau khi có tokenizer

    # Tokenizer & Model
    model_name = "vinai/phobert-base" if args.model_name is None else args.model_name
    print(f"[Config] model={model_name} max_len={args.max_len} bs={args.batch_size} "
          f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
          f"class_weight={args.class_weight} focal={args.focal} "
          f"normalize={args.normalize} prefix={args.prefix} use_seg={args.use_seg} "
          f"val_ratio={args.val_ratio} seed={args.seed} eval_strategy={args.eval_strategy}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    ds_val   = ds_val.map(tok_fn,   batched=True, remove_columns=["text"])
    ds_train.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    ds_val.set_format(type="torch",   columns=["input_ids","attention_mask","labels"])

    if args.test_csv:
        ds_test_raw = dataset_from_dataframe_for_eval(df_test, test_has_label)
        ds_test = ds_test_raw.map(tok_fn, batched=True, remove_columns=["text"])
        if test_has_label:
            ds_test.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        else:
            ds_test.set_format(type="torch", columns=["input_ids","attention_mask"])

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABELS_5),
        id2label={i:l for i,l in enumerate(LABELS_5)},
        label2id={l:i for i,l in enumerate(LABELS_5)},
        hidden_dropout_prob=args.hidden_dropout,
        attention_probs_dropout_prob=args.attn_dropout
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # TrainingArguments
    training_args, have_evalsave = build_training_args(args, len_train=len(ds_train))

    # Trainer init (compat: processing_class vs tokenizer)
    trainer_sig = signature(Trainer.__init__).parameters
    use_processing_class = ("processing_class" in trainer_sig)

    class_weight = compute_class_weight_from_train(ds_train) if args.class_weight else None
    if class_weight is not None and torch.cuda.is_available():
        class_weight = class_weight.cuda()

    # callbacks
    log_csv = os.path.join(args.output_dir, "training_log.csv")
    eval_cb = EvalAndLogCallback(
        log_csv_path=log_csv,
        ds_test=ds_test,
        test_has_label=test_has_label,
        id2lbl=ID2LBL,
        do_manual_eval_per_epoch=not have_evalsave and (args.eval_strategy != "no")
    )
    callbacks_list = [eval_cb]
    if HAS_EARLYSTOP and can_use_early_stopping() and args.eval_strategy != "no":
        callbacks_list.append(EarlyStoppingCallback(early_stopping_patience=2))

    data_collator = DataCollatorWithPadding(
        tokenizer=tok,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=build_metrics(),
        callbacks=callbacks_list,
        data_collator=data_collator
    )
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

    # Gắn trainer vào callback để callback gọi evaluate/predict
    eval_cb.attach_trainer(trainer)

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from if args.resume_from else None)

    # (Giữ lại evaluate cuối để chắc có kết quả tổng kết)
    try:
        eval_out = trainer.evaluate()
        print("\n[Evaluate on VAL] ->", eval_out)
    except Exception as e:
        print("[Evaluate] skipped:", e)

    # Lưu dự đoán VAL cuối
    val_preds = trainer.predict(ds_val)
    y_true = val_preds.label_ids
    y_pred = val_preds.predictions.argmax(-1)
    logits = val_preds.predictions
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
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

    # Save model + tokenizer
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Chạy TEST cuối (tùy chọn) – giữ nguyên như trước để có file test cuối cùng
    if ds_test is not None:
        test_preds = trainer.predict(ds_test)
        pred_labels = test_preds.predictions.argmax(-1)
        logits = test_preds.predictions
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        pmax  = probs.max(axis=1)

        out_csv = os.path.join(args.output_dir, "test_predictions.csv")
        if test_has_label:
            cy = test_preds.label_ids
            print("\n=== TEST (labeled, final) ===")
            print(confusion_matrix(cy, pred_labels))
            print(classification_report(cy, pred_labels, target_names=LABELS_5, digits=4))

            df_out = pd.DataFrame({
                "_true": [ID2LBL[int(i)] for i in cy],
                "_pred": [ID2LBL[int(i)] for i in pred_labels],
                "_pmax": pmax
            })
            df_out.to_csv(out_csv, index=False)
            print("[Saved] Test predictions ->", out_csv)

            df_err = df_out[df_out["_true"] != df_out["_pred"]]
            err_csv = os.path.join(args.output_dir, "test_misclassified.csv")
            df_err.to_csv(err_csv, index=False)
            print("[Saved] Misclassified ->", err_csv)
        else:
            df_out = pd.DataFrame({
                "_pred": [ID2LBL[int(i)] for i in pred_labels],
                "_pmax": pmax
            })
            df_out.to_csv(out_csv, index=False)
            print("[Saved] Test predictions (no label) ->", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="/home/dat/llm_ws/data/train/train.csv")
    p.add_argument("--test_csv",  default="/home/dat/llm_ws/data/test/test.csv")
    p.add_argument("--output_dir", default="phobert_5cls_clean")
    p.add_argument("--model_name", default=None, help="vinai/phobert-base hoặc phobert-large")

    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=64)  # chỉnh theo GPU
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--hidden_dropout", type=float, default=0.2)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # Booleans
    p.add_argument("--class_weight", action="store_true", default=False)
    p.add_argument("--focal", action="store_true", default=False)
    p.add_argument("--normalize", action="store_true", default=False)
    p.add_argument("--prefix", action="store_true", default=False, help="Chỉ áp dụng cho TRAIN")
    p.add_argument("--use_seg", action="store_true", default=False)
    p.add_argument("--no_stratify", action="store_true", default=False)

    # Split control
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # Eval/save control (tùy môi trường: sẽ chỉ áp dụng khi cả hai key support)
    p.add_argument("--eval_strategy", choices=["epoch","steps","no"], default="no",
                   help="Nếu môi trường không hỗ trợ đầy đủ, script sẽ tự emulate mỗi epoch và không dùng load_best_model_at_end.")
    p.add_argument("--eval_steps", type=int, default=200,
                   help="Chỉ dùng khi --eval_strategy=steps")

    # Resume
    p.add_argument("--resume_from", type=str, default=None,
                   help="Đường dẫn checkpoint để tiếp tục train")

    args = p.parse_args()
    main(args)
