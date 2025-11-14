import csv
import json
import argparse
import time
import unicodedata
import re
import requests

# ================== Cấu hình ==================
DEFAULT_MODEL = "qwen3:8b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
LABELS = ["very_positive", "positive", "neutral", "negative", "very_negative"]
DEBUG = False  # Bật True nếu muốn in log

FEW_SHOT = [
    ('"san pham ban phim co hai long, gia cong on; giao nhanh."', 'positive'),
    ('"Dùng văn phòng thì được , vỏ máy khá"', 'neutral'),
    ('"âm thanh tạm , pin chưa ổn , còn lại không nổi bật"', 'neutral'),
    ('"sản phẩm laptop rất kinh khủng, gia công cực kinh khủng; phản hồi chậm, rất bực mình."', 'very_negative'),
    ('"dien thoai gia cao, nhiet do chua tot; dong goi au."', 'negative'),
    ('"máy tính bảng quá tuyệt vời, dịch vụ nổi bật, hỗ trợ tốt; rất đáng tiền."', 'very_positive'),
    ('"Giao hàng khá nhanh , đóng gói không tốt , chất lượng bình thường"', 'neutral'),
]

SYSTEM_PROMPT = """Bạn là bộ phân loại cảm xúc bình luận sản phẩm tiếng Việt.
Nhãn hợp lệ: very_positive, positive, neutral, negative, very_negative.

Yêu cầu:
- Đọc bình luận sản phẩm.
- Chọn MỘT nhãn duy nhất thể hiện cảm xúc tổng thể.
- CHỈ trả về đúng chuỗi nhãn, KHÔNG thêm bất kỳ văn bản nào khác.

Nhãn trả về phải là MỘT TRONG:
very_positive
positive
neutral
negative
very_negative
"""

def build_prompt(user_text: str) -> str:
    """
    Tạo prompt few-shot dạng:
    Bình luận: "..."; Nhãn: <label>
    ...
    Bình luận: "..."; Nhãn:
    """
    lines = []
    for t, lab in FEW_SHOT:
        lines.append(f'Bình luận: {t}')
        lines.append(f'Nhãn: {lab}')
        lines.append("")  # dòng trống cho dễ đọc

    lines.append(f'Bình luận: "{user_text}"')
    lines.append('Nhãn:')
    return "\n".join(lines)

def call_ollama(prompt: str, model_name: str) -> str:
    """
    Gọi Ollama và trả về raw text (response).
    Không dùng JSON, không dùng format đặc biệt.
    """
    payload = {
        "model": model_name,
        "prompt": SYSTEM_PROMPT + "\n\n" + prompt,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        },
        "keep_alive": 0,
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    resp = r.json()
    raw = resp.get("response", "").strip()

    if DEBUG:
        print("========== RAW MODEL OUTPUT ==========")
        print(raw)
        print("======================================")

    return raw

def normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    return s

def extract_label(raw: str) -> str:
    """
    Từ chuỗi raw mà model trả về, cố gắng lôi ra 1 nhãn trong LABELS.
    Ưu tiên:
      1. Nếu chứa nguyên chuỗi nhãn (vd 'very_positive')
      2. Nếu có từ gần giống (vd 'very positive', 'verypositive', 'negative.', ...)
    Nếu không tìm được gì -> neutral.
    """
    if not raw:
        return "neutral"

    s = normalize_str(raw)

    # 1) Nếu raw chứa trực tiếp 1 nhãn chuẩn
    for lab in LABELS:
        if lab in s:
            return lab

    # 2) Map alias đơn giản
    alias_map = {
        "very positive": "very_positive",
        "verypositive": "very_positive",
        "rất tốt": "very_positive",
        "tuyệt vời": "very_positive",

        "positive": "positive",
        "tốt": "positive",
        "good": "positive",

        "neutral": "neutral",
        "trung tính": "neutral",
        "bình thường": "neutral",
        "binh thuong": "neutral",
        "ổn": "neutral",
        "on": "neutral",

        "negative": "negative",
        "negativ": "negative",
        "xấu": "negative",
        "bad": "negative",

        "very negative": "very_negative",
        "verynegative": "very_negative",
        "rất tệ": "very_negative",
        "kinh khủng": "very_negative",
        "tồi tệ": "very_negative",
    }

    # check alias theo substring
    for key, lab in alias_map.items():
        if key in s:
            return lab

    # 3) Thử tách token chữ cái và underscore rồi xét từng token
    tokens = re.findall(r"[a-z_]+", s)
    for tok in tokens:
        t = normalize_str(tok)
        t = t.replace("-", "_")
        t = t.replace(".", "")
        # map alias đơn giản
        if t in alias_map:
            return alias_map[t]
        if t in LABELS:
            return t

    # Nếu bó tay thì cho neutral
    return "neutral"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV vào: có cột 'text' (và tuỳ chọn 'label')")
    ap.add_argument("--out_csv", required=True, help="CSV ra")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    model_name = args.model

    with open(args.input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert "text" in reader.fieldnames, "Thiếu cột 'text' trong CSV"
        rows = list(reader)

    out_rows = []
    total = len(rows)

    for i, row in enumerate(rows, 1):
        txt = row["text"]
        prompt = build_prompt(txt)

        try:
            raw = call_ollama(prompt, model_name)
            label = extract_label(raw)

            if DEBUG:
                print(f"[DEBUG] text={txt!r}")
                print(f"        raw={raw!r}, label={label!r}")

        except Exception as e:
            if DEBUG:
                print(f"[ERROR] row {i}, text={txt!r}, error={e}")
            label = "neutral"

        new_row = dict(row)
        new_row["pred_label"] = label
        out_rows.append(new_row)

        if i % 20 == 0 or i == total:
            print(f"Processed {i}/{total}")

        time.sleep(0.02)

    # Ghi CSV: giữ nguyên các cột cũ + thêm pred_label
    fieldnames = list(rows[0].keys())
    if "pred_label" not in fieldnames:
        fieldnames.append("pred_label")

    with open(args.out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Done. Wrote {len(out_rows)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
