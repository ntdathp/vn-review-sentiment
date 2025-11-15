import json
import csv

INPUT_JSON = "/home/dat/Downloads/datafortrain/HomeLifestyle/HomeLifestyle-dev.json"  # đổi tên file nếu cần
OUTPUT_CSV = "/home/dat/Downloads/datafortrain/HomeLifestyle/HomeLifestyle-dev.csv"

rating_map = {
    5: "very_positive",
    4: "positive",
    3: "neutral",
    2: "negative",
    1: "very_negative",
}

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)    # list các dict

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    # header
    writer.writerow(["text", "rating"])

    for item in data:
        # chỉ giữ những comment có Helpfulness_Score 4 hoặc 5
        hs = item.get("Helpfulness_Score", None)
        try:
            hs = int(hs) if hs is not None else None
        except ValueError:
            hs = None

        if hs not in (4, 5):
            continue

        raw_comment = str(item.get("Comment", ""))
        raw_rating = item.get("Rating", None)

        # Chuẩn hóa xuống dòng -> '. '
        comment = raw_comment.replace("\r\n", "\n").replace("\r", "\n")
        comment = comment.replace("\n", ". ")
        comment = " ".join(comment.split())  # gom khoảng trắng

        # Map rating số -> label
        label = rating_map.get(raw_rating, "")  # nếu lạ thì để rỗng

        writer.writerow([comment, label])

print(f"Đã ghi ra file: {OUTPUT_CSV}")
