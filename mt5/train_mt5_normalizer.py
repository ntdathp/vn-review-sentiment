#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, argparse, unicodedata, re, math
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

# ===================== Teencode / Emoji (cho bẩn hoá & decode constraint) =====================
EMO_POS = ["🤩","🥰","😍","❤️","👍","😎","👌","✨","🔥","💯","❤","♥","💕","💖","💗","💓","💞","💘","💝","💟","😄","😁","😃","🙂","😊","😌","🤗","👏","🙌","🫶","⭐","🌟","⚡","🎉","🥳","🔝","🆒","🆗","✅"]
EMO_NEG = ["😱","😡","🤬","💩","👎","😤","😞","😭","😠","😖","😣","😫","😩","🥵","🥶","🤢","🤮","😷","⚠","❌","🆘","💔","🥲","😓","😔","😕"]

# slang tiêu cực phổ biến (để dirtyfy & option cấm khi decode)
SLANG_NEG = {"vcl", "vkl", "vl"}

TEENCODE_INV: Dict[str, List[str]] = {
    "không": ["ko","kh","k","khong","hong","hông","hok","hem","hêm","hông có","hong co"],
    "chưa": ["chua","chz","chưaaa","ch vẫn"],
    "rất": ["rat","rấttt","vl","vcl","vvcl","cực","cực kỳ","cực kì","ck","max","siêu"],
    "quá": ["qua","wá","qa","qá","quaaaa","vl","vcl"],
    "hơi": ["hoi","h","hơi bị"],
    "ổn": ["on","ổn áp","ok","oke","okela","okla"],

    "có": ["co","c","cóa","có á","y","yes","yep","yup"],
    "được": ["dc","đc","đk","ok","oke","okie","oki","oklah"],
    "rồi": ["r","roi","rùi","r nè","r nha"],
    "đúng": ["dung","chuẩn","chính xác","chuan","chuẩn bài"],

    "tôi": ["t","toi","tui","tớ","mềnh"],
    "mình": ["mk","mik","m","minh","mính"],
    "bạn": ["b","bn","b ơi","bro","cậu"],

    "gì": ["j","ji","cj","cái j"],
    "tại sao": ["ts","vì sao","sao dz","sao z"],
    "như thế nào": ["ntn","như nào"],
    "vậy": ["v","z","dz"],
    "bây giờ": ["bg","bh"],

    "sản phẩm": ["sp","s/phẩm","spham","san pham","sản ph"],
    "đơn hàng": ["đh","don hang","đơn"],
    "khuyến mãi": ["km","sale","gg","flash sale","fs"],
    "giảm giá": ["gg","sale off","down giá"],
    "quảng cáo": ["qc","ads"],
    "bảo hành": ["bh","bao hanh"],
    "chính hãng": ["chh","auth","authen"],

    "giao hàng": ["ship","gh","giao","giao lẹ","ship lẹ"],
    "đóng gói": ["đg","dong goi","pack","package","đóng g"],

    "đáng tiền": ["đáng lắm","xứng đáng"],
    "thất vọng": ["that vong","tv","tụt mood","siêu thất vọng"],
    "tệ": ["te","tệ vl","tệ vcl"],
    "kém": ["kem","dởm"],

    "giao nhanh": ["ship nhanh","giao cấp tốc"],
    "đóng gói cẩn thận": ["đg kĩ","pack kĩ","đóng g kĩ","đóng gói kỹ","pack kỹ"],
    "đúng mô tả": ["đúng như mô tả","đúng như hình"],
    "không như mô tả": ["không đúng mô tả","không giống hình","ko như mta"],

    # vài cụm review “đích” để dirtyfy ra “g ẩu”, “g ảo”, …
    "đóng gói sơ sài": ["đóng ẩu","gói ẩu","đóng g ẩu","g ẩu"],
}

# ===================== Utils cho dirtyfy =====================
def nfc(s): return unicodedata.normalize("NFC", s)

def strip_accents_simple(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ","d").replace("Đ","D")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def maybe_drop_diacritics(sent: str, p=0.5):
    return strip_accents_simple(sent) if random.random() < p else sent

def apply_teencode_phrases(sent: str, prob=0.35):
    s = sent
    if random.random() > prob: return s
    keys = sorted([k for k in TEENCODE_INV if " " in k], key=lambda k: -len(k))
    for k in keys:
        if k in s and random.random() < 0.6:
            s = s.replace(k, random.choice(TEENCODE_INV[k]))
    return s

def apply_teencode_words(sent: str, prob=0.6):
    if random.random() > prob: return sent
    toks = sent.split()
    for i,w in enumerate(toks):
        lw = w.lower()
        if lw in TEENCODE_INV and " " not in lw:
            repl = random.choice(TEENCODE_INV[lw])
            if w.isupper(): repl = repl.upper()
            elif w[0].isupper(): repl = repl[0].upper() + repl[1:]
            toks[i] = repl
    return " ".join(toks)

def maybe_insert_emojis(sent: str, prob=0.5):
    if random.random() > prob: return sent
    if re.search(r"(tuyệt vời|đỉnh|rất thích|hài lòng)", sent, flags=re.I): return sent + " " + random.choice(EMO_POS)
    if re.search(r"(tệ|kém|thất vọng|không hài lòng|xấu|sơ sài)", sent, flags=re.I): return sent + " " + random.choice(EMO_NEG)
    return sent

def dirtyfy(clean: str) -> str:
    s = clean
    s = maybe_drop_diacritics(s, 0.55)
    s = apply_teencode_phrases(s, 0.35)
    s = apply_teencode_words(s, 0.6)
    # thỉnh thoảng trộn slang tiêu cực
    if random.random() < 0.3:
        s += " " + random.choice(list(SLANG_NEG))
    s = maybe_insert_emojis(s, 0.5)
    if random.random() < 0.6: s = s.lower()
    return nfc(re.sub(r"\s+", " ", s).strip())

# ===================== Synthetic từ template canonical =====================
BUILTIN_TEMPLATES = [
    # chất lượng / logistic / tiêu cực
    "Sản phẩm đóng gói sơ sài, tôi rất thất vọng.",
    "Sản phẩm không như mô tả.",
    "Đóng gói cẩn thận, giao nhanh.",
    "Sản phẩm kém chất lượng.",
    "Tôi không hài lòng về đóng gói.",
    "Sản phẩm đáng tiền.",
    "Sản phẩm quá tệ.",
    "Sản phẩm chính hãng.",
    "Dịch vụ giao hàng chậm.",
    "Tôi muốn đổi trả sản phẩm.",
]

def build_synth_from_templates(templates: List[str], variants_per: int = 6) -> DatasetDict:
    src, tgt = [], []
    for t in templates:
        tgt.append(t)
        src.append(t)  # bản sạch gốc
        for _ in range(variants_per):
            src.append(dirtyfy(t))
            tgt.append(t)
    data = {"src": src, "tgt": tgt}
    ds = Dataset.from_dict(data)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return DatasetDict(train=ds["train"], validation=ds["test"])

# ===================== Metric đơn giản =====================
def chrF1(ref: str, hyp: str) -> float:
    ref_set, hyp_set = set(ref), set(hyp)
    inter = len(ref_set & hyp_set)
    if inter == 0: return 0.0
    p = inter / max(1, len(hyp_set)); r = inter / max(1, len(ref_set))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

@dataclass
class DataArgs:
    model_name: str = "google/mt5-small"
    max_src_len: int = 96
    max_tgt_len: int = 96

# ===================== LogitsProcessor chặn chuỗi con =====================
class ForbidSubstringsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, banned_substrings: List[str]):
        self.tok = tokenizer
        # chuẩn hoá NFC để so trùng chắc chắn
        self.banned = [unicodedata.normalize("NFC", s) for s in banned_substrings]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (num_beams, cur_len), scores: (num_beams, vocab_size)
        vocab_size = scores.shape[-1]
        # lặp qua từng beam
        for i in range(input_ids.size(0)):
            prefix = unicodedata.normalize("NFC", self.tok.decode(input_ids[i], skip_special_tokens=True))
            # thử cộng từng token ứng viên, nếu tạo ra chuỗi bị cấm → -inf
            # lưu ý: có thể tối ưu top-k; để đơn giản cứ quét hết vocab
            for tid in range(vocab_size):
                if torch.isneginf(scores[i, tid]):
                    continue
                piece = self.tok.decode([tid], skip_special_tokens=True)
                if not piece:
                    continue
                candidate = unicodedata.normalize("NFC", prefix + piece)
                if any(bad in candidate for bad in self.banned):
                    scores[i, tid] = float("-inf")
        return scores

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","infer"], default="train")
    ap.add_argument("--model_name", default="google/mt5-small")
    ap.add_argument("--output_dir", required=True)

    # dữ liệu
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--dev_csv",   default=None)
    ap.add_argument("--split_ratio", type=float, default=0.1)
    ap.add_argument("--train_jsonl", default=None)
    ap.add_argument("--clean_txt", default=None)
    ap.add_argument("--num_variants", type=int, default=3)

    # bật synthetic từ template canonical
    ap.add_argument("--use_builtin_templates", action="store_true",
                    help="Trộn thêm synthetic từ các câu canonical built-in.")
    ap.add_argument("--template_variants_per", type=int, default=6)

    # train hparams
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=2)

    # infer
    ap.add_argument("--predict", default=None)
    ap.add_argument("--gen_beams", type=int, default=6)
    ap.add_argument("--gen_maxlen", type=int, default=96)
    ap.add_argument("--gen_minlen", type=int, default=0)
    ap.add_argument("--no_repeat_ngram", type=int, default=4)
    ap.add_argument("--length_penalty", type=float, default=1.1)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--forbid_slang_at_decode", action="store_true",
                    help="Cấm vcl/vkl/vl, emoji và <extra_id_*> khi generate (decode constraint, không hậu xử lý).")

    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    # ===== tokenizer/model =====
    def load_tok_mdl(name_or_dir: str):
        tok = AutoTokenizer.from_pretrained(name_or_dir, use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name_or_dir)
        return tok, mdl

    # ===================== INFER =====================
    if args.mode == "infer":
        model_dir = args.output_dir if os.path.isdir(args.output_dir) else args.model_name
        tok, mdl = load_tok_mdl(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device).eval()
        if not args.predict:
            raise SystemExit("Cần --predict \"chuỗi cần chuẩn hoá\"")

        enc = tok([args.predict], truncation=True, max_length=args.gen_maxlen, return_tensors="pt").to(device)

        # --- Cấu hình chặn ---
        bad_words_ids = None
        logits_processors = None

        if args.forbid_slang_at_decode:
            bad_words_ids = []
            # 1) Chặn toàn bộ sentinel tokens <extra_id_0..99>
            for i in range(100):
                toks = tok.encode(f"<extra_id_{i}>", add_special_tokens=False)
                if toks:
                    bad_words_ids.append(toks)

            # 2) Chặn slang theo chuỗi token (nếu tokenizer ghép được nguyên chuỗi)
            for bad in sorted(SLANG_NEG):
                toks = tok.encode(bad, add_special_tokens=False)
                if toks:
                    bad_words_ids.append(toks)

            # 3) Chặn emoji ở mức token (nhiều emoji là 1 token)
            for emo in EMO_POS + EMO_NEG:
                toks = tok.encode(emo, add_special_tokens=False)
                if toks:
                    bad_words_ids.append(toks)

            # 4) Chặn theo chuỗi con (mạnh nhất)
            banned_substrings = set()
            # slang + các biến thể sát nghĩa
            for s in ["vcl", "vkl", "vl"]:
                banned_substrings.add(s)
                banned_substrings.add(" " + s)
                banned_substrings.add(s + " ")
                banned_substrings.add(" v kl")  # một biến thể hay gặp

            # emoji (nếu muốn output sạch không emoji)
            for emo in EMO_POS + EMO_NEG:
                banned_substrings.add(emo)
                banned_substrings.add(" " + emo)

            logits_processors = LogitsProcessorList([
                ForbidSubstringsProcessor(tok, list(banned_substrings))
            ])

        with torch.no_grad():
            out = mdl.generate(
                **enc,
                max_length=args.gen_maxlen,
                min_length=args.gen_minlen,
                num_beams=args.gen_beams,
                no_repeat_ngram_size=args.no_repeat_ngram,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                early_stopping=True,
                bad_words_ids=bad_words_ids,
                logits_processor=logits_processors,
                # forced_eos_token_id=tok.eos_token_id,  # mở nếu muốn chắc chắn kết thúc bằng EOS
            )
        print("[PRED]", tok.decode(out[0], skip_special_tokens=True).strip())
        return

    # ===================== TRAIN =====================
    # 1) dữ liệu từ CSV/JSONL/TXT sạch
    if args.train_csv:
        if args.dev_csv:
            ds = load_dataset("csv", data_files={"train": args.train_csv, "validation": args.dev_csv})
        else:
            tmp = load_dataset("csv", data_files={"full": args.train_csv})["full"]
            split = tmp.train_test_split(test_size=args.split_ratio, seed=args.seed)
            ds = DatasetDict(train=split["train"], validation=split["test"])

        def _clean(b):
            return {"src": [nfc(str(x).strip()) for x in b["src"]],
                    "tgt": [nfc(str(x).strip()) for x in b["tgt"]]}
        for split_name in ds.keys():
            cols = set(ds[split_name].column_names)
            if not {"src","tgt"}.issubset(cols):
                raise ValueError(f"[{split_name}] CSV cần cột 'src' & 'tgt', thấy: {cols}")
        ds = ds.map(_clean, batched=True)

    elif args.train_jsonl:
        def _gen(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    yield {"src": nfc(j["src"]).strip(), "tgt": nfc(j["tgt"]).strip()}
        items = list(_gen(args.train_jsonl))
        cut = int(0.9 * len(items)) if len(items) > 1 else 1
        ds = DatasetDict(
            train=Dataset.from_list(items[:cut]),
            validation=Dataset.from_list(items[cut:])
        )

    elif args.clean_txt:
        # clean_txt -> dirtyfy để tạo cặp (src bẩn, tgt sạch)
        src, tgt = [], []
        with open(args.clean_txt, "r", encoding="utf-8") as f:
            lines = [nfc(x.strip()) for x in f if x.strip()]
        for line in lines:
            for _ in range(args.num_variants):
                src.append(dirtyfy(line)); tgt.append(line)
        base = Dataset.from_dict({"src": src, "tgt": tgt})
        base = base.train_test_split(test_size=0.1, seed=args.seed)
        ds = DatasetDict(train=base["train"], validation=base["test"])

    else:
        default_train = "/home/dat/llm_ws/data/train/mt5_norm_train.csv"
        default_dev   = "/home/dat/llm_ws/data/train/mt5_norm_dev.csv"
        if os.path.exists(default_train) and os.path.exists(default_dev):
            ds = load_dataset("csv", data_files={"train": default_train, "validation": default_dev})
        else:
            raise ValueError("Cần --train_csv [--dev_csv] hoặc --train_jsonl hoặc --clean_txt.")

    # 2) trộn thêm synthetic từ template canonical (rất quan trọng)
    if args.use_builtin_templates:
        synth = build_synth_from_templates(BUILTIN_TEMPLATES, variants_per=args.template_variants_per)
        # concat
        train = Dataset.from_dict({
            "src": list(ds["train"]["src"]) + list(synth["train"]["src"]),
            "tgt": list(ds["train"]["tgt"]) + list(synth["train"]["tgt"]),
        })
        valid = Dataset.from_dict({
            "src": list(ds["validation"]["src"]) + list(synth["validation"]["src"]),
            "tgt": list(ds["validation"]["tgt"]) + list(synth["validation"]["tgt"]),
        })
        ds = DatasetDict(train=train, validation=valid)

    # 3) tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tok.padding_side = "right"
    mdl = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if args.gradient_checkpointing:
        mdl.gradient_checkpointing_enable()
        mdl.config.use_cache = False

    data_args = DataArgs(model_name=args.model_name)

    def preprocess(batch):
        model_inputs = tok(batch["src"], max_length=data_args.max_src_len,
                           truncation=True, padding="max_length")
        labels = tok(text_target=batch["tgt"], max_length=data_args.max_tgt_len,
                     truncation=True, padding="max_length")["input_ids"]
        pad_id = tok.pad_token_id
        labels = [[(t if t != pad_id else -100) for t in seq] for seq in labels]
        model_inputs["labels"] = labels
        return model_inputs

    ds_tok = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    # sanity
    if all(t == -100 for t in ds_tok["train"][0]["labels"]):
        raise RuntimeError("Label toàn -100 → tiền xử lý sai.")

    collator = DataCollatorForSeq2Seq(tok, model=mdl, label_pad_token_id=-100)

    # preflight loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).train()
    try:
        batch = {k: (torch.tensor(v[:2]).to(device) if isinstance(v, list) else v[:2].to(device))
                 for k, v in {k: ds_tok["train"][k] for k in ["input_ids","attention_mask","labels"]}.items()}
        with torch.no_grad():
            out32 = mdl(**batch)
            base_loss = float(out32.loss.detach())
        print("[CHECK] float32 loss:", base_loss)
    except Exception as e:
        print("[WARN] Bỏ qua preflight loss:", e)

    # train args
    try:
        args_train = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            lr_scheduler_type="linear",
            warmup_ratio=0.05,
            gradient_accumulation_steps=args.grad_accum,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            predict_with_generate=True,
            generation_max_length=data_args.max_tgt_len,
            logging_steps=50,
            save_total_limit=3,
            fp16=args.fp16,
            bf16=args.bf16,
            max_grad_norm=1.0,
            label_smoothing_factor=0.0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_chrF",
            greater_is_better=True,
            seed=args.seed,
            report_to=[],
            gradient_checkpointing=args.gradient_checkpointing,
            optim="adafactor",
        )
    except TypeError:
        print("[WARN] transformers cũ → cấu hình rút gọn.")
        args_train = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            logging_steps=50,
            save_total_limit=3,
            fp16=args.fp16,
            bf16=args.bf16,
            max_grad_norm=1.0,
            label_smoothing_factor=0.0,
            seed=args.seed,
            report_to=[],
            optim="adafactor",
        )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        pred_str = tok.batch_decode(preds, skip_special_tokens=True)
        label_str = tok.batch_decode(labels, skip_special_tokens=True)
        scores = [chrF1(r, p) for r, p in zip(label_str, pred_str)]
        return {"chrF": float(np.mean(scores))}

    trainer = Seq2SeqTrainer(
        model=mdl,
        args=args_train,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
