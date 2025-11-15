#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Toolbox GUI (PyQt6) — Auto-load my_phobert_only.py, toggle PhoBERT <-> LLM (HTTP)
V2.7-patched — Nới gate để tránh skip vô lý các review thật (e-commerce, mỹ phẩm…)
  • Thêm nhận diện "review thật" sớm: có ≥2 từ nội dung (≥4 ký tự, có nguyên âm) + từ khóa e-commerce → KHÔNG skip
  • Không skip nếu có cụm vận chuyển/đóng gói/thương hiệu/phẩm chất ("giao hàng", "đóng gói", "chính hãng", "mùi hương", "dưỡng ẩm"…)
  • Hạ độ nhạy các rule gây oan: 
      - repeated-short-token: cần ≥4 lần (trước đây 3) và token chiếm ≥60% tổng token
      - many-very-short: cần ≥4 token (trước đây 3)
      - too-short-words: chỉ áp dụng khi tok_cnt ≤3 (trước đây ≤4) và không có hint/lexicon
      - multi-token-gibberish: nới ngưỡng và bỏ qua nếu có pos/neg signal hay cụm e-commerce
  • Vẫn skip các chuỗi lặp vô nghĩa kiểu "hài lòng?" * 30, hoặc spam ký tự, hoặc tục tĩu
"""
from __future__ import annotations

import sys, os, traceback, importlib, importlib.util, html, json, re, unicodedata, string, math
from typing import Callable, Optional
from collections import Counter
from PyQt6 import QtCore, QtWidgets, QtGui

# ====== import textproc (file bạn tự code) ======
try:
    import textproc as tp
except Exception:
    tp = None  # vẫn chạy, chỉ mất 1 số tín hiệu

# Optional: HTTP client for LLM
try:
    import requests
except Exception:
    requests = None

# Colors
YOU_COLOR   = "#6a1b9a"
MODEL_COLOR = "#1565c0"
INFO_COLOR  = "#2e7d32"
ERROR_COLOR = "#c62828"
TEXT_COLOR  = "#000000"

# -------------------------------
# Dynamic model loader
# -------------------------------
class ModelClient(QtCore.QObject):
    modelLoaded = QtCore.pyqtSignal(str)
    modelCleared = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._infer_fn: Optional[Callable[[str], str]] = None
        self._normalize_fn: Optional[Callable[[str], str]] = None
        self._module_name: Optional[str] = None
    def _load_from_module(self, module) -> None:
        infer_fn = getattr(module, "infer", None)
        if not callable(infer_fn):
            raise AttributeError("Module không định nghĩa hàm infer(text: str) -> str.")
        normalize_fn = getattr(module, "normalize", None)
        if normalize_fn is not None and not callable(normalize_fn):
            normalize_fn = None
        self._infer_fn = infer_fn
        self._normalize_fn = normalize_fn
    def load_from_module_name(self, name: str):
        module = importlib.import_module(name)
        self._load_from_module(module)
        self._module_name = name
        self.modelLoaded.emit(name)
    def clear(self):
        self._infer_fn = None
        self._normalize_fn = None
        self._module_name = None
        self.modelCleared.emit()
    def has_infer(self) -> bool:
        return callable(self._infer_fn)
    def has_normalize(self) -> bool:
        return callable(self._normalize_fn)
    def infer(self, text: str, use_normalize: bool = False) -> str:
        if not self.has_infer():
            return f"[echo] {text}"
        if use_normalize and self.has_normalize():
            try:
                text = self._normalize_fn(text)  # type: ignore[misc]
            except Exception:
                traceback.print_exc()
        return self._infer_fn(text)  # type: ignore[misc]

# -------------------------------
# Generic worker
# -------------------------------
class InferWorker(QtCore.QObject):
    started = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str, str)
    failed = QtCore.pyqtSignal(str)
    def __init__(self, runner: Callable[[str], str], text: str, parent=None):
        super().__init__(parent); self._runner = runner; self._text = text
    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.started.emit(self._text)
            out = self._runner(self._text)
            if not isinstance(out, str): out = str(out)
            self.finished.emit(self._text, out)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")

# -------------------------------
# Main Window
# -------------------------------
class ChatWindow(QtWidgets.QMainWindow):
    # ===== Regex / dicts for low-info gate =====
    _VI_NONWORD = r"[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]"
    _PAT_ALO = re.compile(r"^\s*(alo[\s!?.]*){1,10}$", re.IGNORECASE)
    _PAT_REPEAT_SYL = re.compile(r"^(\w{1,6})(?:\s*\1){2,}$", re.UNICODE)  # ≥3 lần tổng

    _PROF_SET = {"cm","cmm","cml","dm","đm","vl","vkl","cc","wtf","lol","shit","fuck","địt","lồn","cặc","đéo","mẹ"}
    _PROF_RE  = re.compile(r"^(?:c+\.?m+(?:\.?l+)?|d+\.?m+|đ+\.?m+)$", re.IGNORECASE | re.UNICODE)

    # Hints: NGUYÊN TỪ (full-word) — không dùng substring
    _HINT_WORDS = {
        # danh mục / hàng hoá
        "sản","phẩm","sản phẩm","san","pham","sanpham","hàng","hang","shop","sp","đơn","don","order",
        # thuộc tính / linh kiện
        "pin","sạc","sac","màn","hình","man","hinh","loa","tai","nghe","âm","bass","wifi","bluetooth",
        "ốp","op","miếng","dán","mieng","dan","áo","quần","giày","dép","ao","quan","giay","dep",
        # shorthands neutral
        "bt","bth",
        # mỹ phẩm phổ thông
        "son","môi","mui","mùi","dưỡng","duong","kem","lip","lipice","dhc","innisfree","cocoon"
    }

    # Short sentiment tokens (có & không dấu)
    _SHORT_POS = {"tốt","tot","ok","oke","oki","ổn","on","đẹp","dep","xịn","xin","ung","ưng"}
    _SHORT_NEG = {"tệ","te","xấu","xau","kém","kem","đắt","dat","phèn","phen","chan","chán","dở","dởm","lởm","dom","lom"}
    _SHORT_NEU = {"bt","bth"}

    # E-commerce review keywords (whitelist)
    _REVIEW_PHRASES = [
        r"giao\s*hàng", r"đóng\s*gói", r"chính\s*hãng", r"đúng\s*mô\s*tả",
        r"mùi\s*hương", r"thơm", r"dưỡng\s*ẩm", r"mịn\s*môi", r"lên\s*màu",
        r"tem\s*chống\s*hàng\s*giả", r"shipper", r"giá\s*(tốt|ổn|ok|hợp\s*lý|rẻ)",
        r"mua\s*lại|lần\s*2|lần\s*nữa", r"bao\s*bì|niêm\s*phong|seal",
    ]
    _REVIEW_RE = [re.compile(pat, re.IGNORECASE | re.UNICODE) for pat in _REVIEW_PHRASES]

    # Chữ cái
    _CONSONANTS = set(chr(c) for c in range(97,123)) - set("aeiouy"); _CONSONANTS.update(list("đ"))
    _VOWELS = set("aeiouy")
    _VI_VOWELS_ALL = set("aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")  # để bắt nguyên âm có dấu
    _RARE_LATINS = set("fjwz")  # TV ít dùng

    # Fallback tích cực/tiêu cực ngắn (mẫu câu)
    _LOCAL_POS_FALLBACK = [
        re.compile(r"\bxịn\s*[sx][òo]\b", re.IGNORECASE),
        re.compile(r"\bquá\s*xịn\b", re.IGNORECASE),
        re.compile(r"\bquá\s*ok(?:e+)?\b", re.IGNORECASE),
        re.compile(r"\bshop\s*(ổn|ok|oke|uy\s*tín|nhiệt\s*tình)\b", re.IGNORECASE),
    ]
    _LOCAL_NEG_FALLBACK = [
        re.compile(r"\b(tệ|xấu|kém|dởm|lởm|đắt|phèn)\b", re.IGNORECASE),
        re.compile(r"\b(sản\s*phẩm|sp|hàng|shop)\s+(rất\s+)?(tệ|xấu|kém|dởm|lởm|đắt|phèn)\b", re.IGNORECASE),
    ]

    # Generic tokens & function words (không dấu)
    _GENERIC_TOKENS = {
        "san","pham","sanpham","hang","don","donhang","shop","sp","mh","item","items","order",
        "hanghoa","hang_hoa","san_pham","san-pham"
    }
    _FUNC_WORDS = {"nhu","la","thi","va","hoac","cua","và","là"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat Toolbox (PyQt6)")
        self.resize(1100, 680)
        self.setMinimumWidth(960)

        self.model = ModelClient(self)
        self.use_llm = False

        # ====== UI ======
        top_bar = QtWidgets.QHBoxLayout(); top_bar.setSpacing(8)
        self.btn_engine = QtWidgets.QPushButton("Engine: PhoBERT")
        self.btn_engine.setCheckable(True); self.btn_engine.setMinimumWidth(160)
        self.btn_engine.setStyleSheet("font-weight:600; padding:6px 10px;")
        self.btn_engine.setToolTip("Đang dùng PhoBERT. Bấm để chuyển sang LLM (HTTP).")

        self.llm_opts = QtWidgets.QFrame(); self.llm_opts.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        llm_layout = QtWidgets.QHBoxLayout(self.llm_opts); llm_layout.setContentsMargins(0,0,0,0); llm_layout.setSpacing(6)
        self.lbl_llm_model = QtWidgets.QLabel("Model:")
        self.cmb_llm_model = QtWidgets.QComboBox(); self.cmb_llm_model.setEditable(True)
        self.cmb_llm_model.addItems(["qwen3:8b","qwen2.5:7b-instruct","qwen2.5:14b-instruct"])
        self.cmb_llm_model.setCurrentText("qwen3:8b"); self.cmb_llm_model.setMinimumWidth(180)
        self.lbl_llm_url = QtWidgets.QLabel("URL:")
        self.txt_llm_url = QtWidgets.QLineEdit("http://localhost:11434/api/generate")
        self.txt_llm_url.setMinimumWidth(260); self.txt_llm_url.setPlaceholderText("http://<host>:<port>/api/generate")
        llm_layout.addWidget(self.lbl_llm_model); llm_layout.addWidget(self.cmb_llm_model)
        llm_layout.addWidget(self.lbl_llm_url); llm_layout.addWidget(self.txt_llm_url)
        self.llm_opts.hide()

        self.lbl_model = QtWidgets.QLabel("No model loaded"); self.lbl_model.setStyleSheet("color: gray;")

        top_bar.addWidget(self.btn_engine); top_bar.addWidget(self.llm_opts); top_bar.addStretch(1); top_bar.addWidget(self.lbl_model)

        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central); main_layout.setContentsMargins(10,10,10,10); main_layout.setSpacing(8)
        main_layout.addLayout(top_bar)

        self.history = QtWidgets.QTextEdit(); self.history.setReadOnly(True)
        self.history.setPlaceholderText("Lịch sử hội thoại sẽ hiển thị ở đây…")
        self.history.setMinimumHeight(320); main_layout.addWidget(self.history, 1)

        input_row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText("Nhập tin nhắn… (Enter để gửi, Shift+Enter để xuống dòng)")
        self.input.installEventFilter(self)
        self.btn_send = QtWidgets.QPushButton("Send"); self.btn_send.setDefault(True)
        self.btn_send.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        input_row.addWidget(self.input, 1); input_row.addWidget(self.btn_send); main_layout.addLayout(input_row)

        bottom_bar = QtWidgets.QHBoxLayout(); bottom_bar.setSpacing(8); bottom_bar.setContentsMargins(0,0,0,0)
        self.btn_save = QtWidgets.QPushButton("Save Log…"); self.btn_clear_chat = QtWidgets.QPushButton("Clear Chat")
        bottom_bar.addWidget(self.btn_save); bottom_bar.addWidget(self.btn_clear_chat); bottom_bar.addStretch(1)
        self.status = QtWidgets.QLabel("Ready"); self.status.setStyleSheet("color: gray;"); self.status.setFixedWidth(90)
        bottom_bar.addWidget(self.status); main_layout.addLayout(bottom_bar)

        # Signals
        self.btn_send.clicked.connect(self.on_send)
        self.btn_save.clicked.connect(self.on_save_log)
        self.btn_clear_chat.clicked.connect(self.on_clear_chat)
        self.model.modelLoaded.connect(self.on_model_loaded)
        self.model.modelCleared.connect(self.on_model_cleared)
        self.btn_engine.toggled.connect(self._on_engine_toggled)

        self._active_threads: list[QtCore.QThread] = []

        # Auto-load model module
        mod_name = os.environ.get("CHAT_TOOLBOX_PHOBERT_MODULE", "my_phobert_only")
        try:
            self.model.load_from_module_name(mod_name)
        except Exception as e:
            self.append_error(f"Không thể auto-load module '{mod_name}': {e}")
        print(">>> Chat Toolbox started. Engine: PhoBERT (default).")

        if tp is None:
            self.append_error("Không import được textproc.py — gate vẫn chạy nhưng không dùng được normalize/lexicon từ textproc.")

    # ---------- UI helpers ----------
    def _append_block(self, title_html: str, body_text: str):
        safe_body = html.escape(body_text)
        block = f"{title_html}<br><span style=\"color:{TEXT_COLOR}; white-space:pre-wrap;\">{safe_body}</span><br><br>"
        self.history.append(block)
    def append_user(self, text: str):
        self._append_block(f'<span style="color:{YOU_COLOR}; font-weight:600;">You:</span>', text)
    def append_model(self, text: str):
        self._append_block(f'<span style="color:{MODEL_COLOR}; font-weight:600;">Model:</span>', text)
    def append_info(self, text: str):
        self._append_block(f'<span style="color:{INFO_COLOR}; font-weight:600;">[Info]</span>', text)
    def append_error(self, text: str):
        self._append_block(f'<span style="color:{ERROR_COLOR}; font-weight:600;">[Error]</span>', text)

    # ---------- Helpers ----------
    def _canon_token(self, tok: str) -> str:
        return re.sub(r'(.)\1{2,}', r'\1\1', tok.lower())

    def _token_is_profanity(self, tok: str) -> bool:
        return (tok in self._PROF_SET) or bool(self._PROF_RE.match(tok))

    def _strip_diacritics_basic(self, s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        s = s.replace("đ","d").replace("Đ","D")
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", s)

    def _consonant_ratio(self, s: str) -> float:
        if not s: return 0.0
        letters = [c for c in s.lower() if c.isalpha()]
        if not letters: return 0.0
        consonants = sum(1 for c in letters if (c in self._CONSONANTS))
        return consonants / len(letters)

    def _vowel_ratio(self, s: str) -> float:
        if not s: return 0.0
        letters = [c for c in s.lower() if c.isalpha()]
        if not letters: return 0.0
        nod = self._strip_diacritics_basic("".join(letters)).lower()
        vowels = sum(1 for c in nod if c in set("aeiouy"))
        return vowels / len(letters)

    def _has_repeating_chunk(self, s: str) -> bool:
        if len(s) < 6: return False
        for k in (2, 3, 4):
            m = re.search(rf"([a-z0-9]{{{k}}})\1+", s.lower())
            if m: return True
        return False

    def _is_vowel_only_short(self, s: str) -> bool:
        s = s.strip(string.punctuation)
        if not s:
            return False
        letters = [ch for ch in s.lower() if ch.isalpha()]
        if not letters or len(letters) > 3:
            return False
        # True nếu tất cả là nguyên âm (kể cả có dấu)
        return all(ch in self._VI_VOWELS_ALL for ch in letters)

    def _norm_words(self, text: str):
        nod = (tp.strip_accents_simple(text) if tp else self._strip_diacritics_basic(text))
        return re.findall(r"[A-Za-zÀ-ỹà-ỹ0-9]+", nod)

    def _is_product_hint_token(self, tok: str) -> bool:
        t = tok.strip().lower()
        if not t: return False
        t = t.strip(string.punctuation)
        if not t: return False
        t_nod = self._strip_diacritics_basic(t)
        return (t in self._HINT_WORDS) or (t_nod in self._HINT_WORDS)

    def _is_short_sentiment_token(self, tok: str) -> bool:
        t = tok.strip().lower()
        if not t: return False
        t = t.strip(string.punctuation)
        if not t: return False
        t_nod = self._strip_diacritics_basic(t)
        return (t in self._SHORT_POS or t in self._SHORT_NEG or t in self._SHORT_NEU
                or t_nod in self._SHORT_POS or t_nod in self._SHORT_NEG or t_nod in self._SHORT_NEU)

    def _has_vi_vowel(self, s: str) -> bool:
        return any(ch.lower() in self._VI_VOWELS_ALL for ch in s)

    def _contains_review_phrase(self, text: str) -> bool:
        return any(p.search(text) for p in self._REVIEW_RE)

    def _content_tokens(self, text: str) -> int:
        words = re.findall(r"[A-Za-zÀ-ỹà-ỹ0-9]+", text, re.UNICODE)
        cnt = 0
        for w in words:
            if len(w) >= 4 and self._has_vi_vowel(w) and not self._token_is_gibberish(w):
                cnt += 1
        return cnt

    def _looks_like_real_review(self, text: str) -> bool:
        # Có ít nhất 2 từ nội dung + cụm e-commerce phổ biến
        if len(text) >= 15 and self._content_tokens(text) >= 2 and self._contains_review_phrase(text.lower()):
            return True
        # Hoặc có ≥3 câu ngắn cách nhau bằng dấu chấm, mỗi câu có từ ≥4 ký tự có nguyên âm
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) >= 3:
            good = sum(any(len(w) >= 4 and self._has_vi_vowel(w) for w in re.findall(r"\w+", s)) for s in sentences)
            if good >= 2:
                return True
        return False

    def _token_is_gibberish(self, t: str) -> bool:
        if not t: return True
        core = t.strip(string.punctuation)
        if not core: return True

        letters = sum(ch.isalpha() for ch in core)
        digits  = sum(ch.isdigit() for ch in core)
        if digits > 0 and letters < 3 and len(core) >= 5:
            return True

        nod = (tp.strip_accents_simple(core) if tp else self._strip_diacritics_basic(core))
        cons_ratio = self._consonant_ratio(nod); vow_ratio = self._vowel_ratio(nod)

        # token 1–2 ký tự: nếu không có nguyên âm & không phải hint/sentiment => rác
        if 1 <= len(core) <= 2:
            if (not self._has_vi_vowel(core)) and (not self._is_product_hint_token(core)) and (not self._is_short_sentiment_token(core)):
                return True

        # token 3–4 ký tự, vowel_ratio rất thấp và không phải hint/sentiment -> rác
        if 3 <= len(nod) <= 4 and vow_ratio <= 0.25:
            if not (self._is_product_hint_token(core) or self._is_short_sentiment_token(core)):
                return True

        # token dài có chữ hiếm (f/j/w/z) và không phải hint/sentiment -> rác
        if len(nod) >= 5 and any(c in self._RARE_LATINS for c in nod):
            if not (self._is_product_hint_token(core) or self._is_short_sentiment_token(core)):
                return True

        # chữ 'q' riêng: nếu không phải "qu" ở đầu => nghi rác khi token ngắn
        if len(nod) <= 4 and "q" in nod and not nod.startswith("qu"):
            if not (self._is_product_hint_token(core) or self._is_short_sentiment_token(core)):
                return True

        # 3+ phụ âm liên tiếp → nghi rác
        if re.search(r"[^aeiouy\d]{3,}", nod) and len(nod) >= 5:
            return True

        # lặp chunk / quá nhiều phụ âm / quá ít nguyên âm
        if len(nod) >= 5 and (self._has_repeating_chunk(nod) or cons_ratio >= 0.75 or vow_ratio <= 0.18):
            return True

        # không là hint và nguyên âm cực thấp
        if len(nod) >= 6 and (not self._is_product_hint_token(core)) and vow_ratio <= 0.15:
            return True

        return False

    # Tautology & generic-only
    def _is_tautology_like(self, text: str) -> bool:
        nod = (tp.strip_accents_simple(text) if tp else self._strip_diacritics_basic(text)).lower()
        nod = re.sub(r"\s+", " ", nod).strip()
        if re.search(r"\b([a-z0-9]{2,}(?:\s+[a-z0-9]{2,}){0,3})\s+(nhu|la|thi)\s+\1\b", nod):
            return True
        toks = re.findall(r"[a-z0-9]+", nod)
        if toks and all(t in self._GENERIC_TOKENS or t in self._FUNC_WORDS for t in toks):
            return True
        return False

    # Generic + Gibberish (câu ngắn)
    def _is_generic_plus_gibberish(self, text: str) -> bool:
        toks = self._norm_words(text)
        if not toks: return False
        generic_or_func = [t for t in toks if (t in self._GENERIC_TOKENS or t in self._FUNC_WORDS)]
        others = [t for t in toks if t not in self._GENERIC_TOKENS and t not in self._FUNC_WORDS]
        gib_cnt = sum(self._token_is_gibberish(x) for x in others if not self._is_product_hint_token(x))
        if gib_cnt >= 1 and len(others) <= 2 and len(generic_or_func) >= max(1, len(toks) - len(others)):
            return True
        return False

    def _sig_stats(self, text: str):
        t = text.strip()
        if not t: return 0.0, 0.0, 1.0, 0, 0, [], [], 0.0, 0.0
        letters = sum(ch.isalpha() for ch in t)
        syms = len(re.findall(self._VI_NONWORD, t))
        alpha_ratio = letters / max(len(t), 1)
        uniq_ratio = len(set(t)) / max(len(t), 1)
        toks = re.findall(r"\w+", t.lower(), re.UNICODE)
        toks_c = [self._canon_token(x) for x in toks]
        tok_cnt = len(toks_c)
        bad_flags = [self._token_is_profanity(x) for x in toks_c]
        bad_tok_cnt = sum(bad_flags)
        sym_ratio = syms / max(len(t), 1)
        avg_len = (sum(len(x) for x in toks_c) / tok_cnt) if tok_cnt else 0.0
        prof_ratio = (bad_tok_cnt / tok_cnt) if tok_cnt else 0.0
        return alpha_ratio, uniq_ratio, sym_ratio, tok_cnt, bad_tok_cnt, toks_c, toks, avg_len, prof_ratio

    def _is_low_info(self, text: str):
        """
        Trả về (True/False, reason_key, reason_msg).
        """
        t = re.sub(r"\s+", " ", text).strip()
        if t == "": return True, "empty", "Văn bản trống."
        if self._PAT_ALO.match(t): return True, "greeting/test-mic", "Chuỗi chào/kiểm tra micro, không đủ ngữ cảnh."
        if self._PAT_REPEAT_SYL.match(t.replace(".", "").replace("!", "").lower()):
            return True, "repeated-syllables", "Chuỗi lặp âm tiết, không chứa ý kiến về sản phẩm."
        if self._is_tautology_like(t):
            return True, "tautology", "Câu lặp lại cùng một cụm (ví dụ: 'X như X'), không chứa nhận xét."

        # ==== NEW: Nhận diện review thật sớm ====
        if self._looks_like_real_review(t):
            return False, "", ""

        # 1 token cực nhiễu
        if self._is_single_token_noise(t):
            return True, "single-token-gibberish", "Văn bản vô nghĩa: 1 từ không có nội dung đánh giá."

        # Generic + Gibberish ngắn
        if self._is_generic_plus_gibberish(t):
            return True, "generic+gibberish", "Chỉ chứa từ chung + 1–2 từ vô nghĩa, không có nhận xét."

        alpha_ratio, uniq_ratio, sym_ratio, tok_cnt, bad_tok_cnt, toks_c, toks, avg_len, prof_ratio = self._sig_stats(t)

        # ===== PRIORITY: nếu có (product-hint) + (short-sentiment) => giữ lại =====
        if 2 <= tok_cnt <= 6:
            has_hint = any(self._is_product_hint_token(w) for w in toks_c)
            has_short_senti = any(self._is_short_sentiment_token(w) for w in toks_c)
            if has_hint and has_short_senti:
                return False, "", ""

        # textproc lexicon signals (nếu có)
        t_norm = tp.normalize_text(t) if tp else t
        try:
            pos_sig, neg_sig = tp.count_lexicon(t_norm) if tp else (0, 0)
        except Exception:
            pos_sig, neg_sig = (0, 0)

        # LOCAL fallback: tích cực & tiêu cực ngắn (pattern)
        if pos_sig == 0 and neg_sig == 0:
            for pat in self._LOCAL_POS_FALLBACK:
                if pat.search(t): pos_sig = 1; break
        if pos_sig == 0 and neg_sig == 0:
            for pat in self._LOCAL_NEG_FALLBACK:
                if pat.search(t): neg_sig = 1; break

        # ===== Bag of noise: sau khi bỏ hint/sentiment không còn token có nguyên âm =====
        word_toks = re.findall(r"[A-Za-zÀ-ỹà-ỹ0-9]+", t, re.UNICODE)
        non_hint2 = [w for w in word_toks if not (self._is_product_hint_token(w) or self._is_short_sentiment_token(w))]
        if non_hint2 and not any(self._has_vi_vowel(w) for w in non_hint2):
            return True, "no-vowel-nonhint", "Không có từ mang nguyên âm/ý nghĩa sau khi loại từ gợi ý."

        # ===== Multi-token gibberish (nới nhẹ): 2–12 token =====
        if 2 <= len(word_toks) <= 12:
            non_hint = [w for w in word_toks
                        if not (self._is_product_hint_token(w) or self._is_short_sentiment_token(w))]
            if non_hint:
                gib_flags    = [self._token_is_gibberish(w) for w in non_hint]
                filler_flags = [self._is_vowel_only_short(w) for w in non_hint]  # ví dụ: "oi", "ai", "ơi"
                gib_cnt      = sum(gib_flags)
                filler_cnt   = sum(filler_flags)
                gib_ratio    = gib_cnt / max(1, len(non_hint))

                need = max(2, math.ceil(0.7 * len(non_hint)))  # trước 0.6 → tăng 0.7 để bớt nhạy

                if (
                    (gib_cnt >= need or gib_ratio >= (3/4) or (gib_cnt >= 2 and filler_cnt >= 1 and len(word_toks) <= 4))
                    and not self._contains_review_phrase(t.lower())
                    and (pos_sig == 0 and neg_sig == 0)
                ):
                    return True, "multi-token-gibberish", "Các từ chủ yếu vô nghĩa, không có nội dung đánh giá."

        # Lặp hạt ngắn (≤3) ≥4 lần *và* chiếm ≥60% token
        short_non_hint = [w.lower() for w in word_toks if (len(w) <= 3 and not self._is_product_hint_token(w) and not self._is_short_sentiment_token(w))]
        if short_non_hint:
            tok_most, n_most = Counter(short_non_hint).most_common(1)[0]
            if n_most >= 4 and n_most >= 0.6 * max(1, len(word_toks)) and (pos_sig == 0 and neg_sig == 0):
                return True, "repeated-short-token", "Lặp từ rất ngắn quá nhiều, thiếu nội dung."

        # ≥4 token rất ngắn (≤2) không có nguyên âm
        very_short_no_vowel = sum(1 for w in word_toks if len(w) <= 2 and not self._has_vi_vowel(w))
        if very_short_no_vowel >= 4 and (pos_sig == 0 and neg_sig == 0):
            return True, "many-very-short", "Quá nhiều từ cực ngắn không có nguyên âm, thiếu nội dung."

        # Repeated token tổng quát: yêu cầu mạnh hơn
        if tok_cnt >= 3:
            tok_most2, n_most2 = Counter([w for w in re.findall(r"\w+", t.lower())]).most_common(1)[0]
            if n_most2 >= 4 and (len(tok_most2) <= 6 or self._token_is_profanity(tok_most2)) and (pos_sig == 0 and neg_sig == 0):
                return True, "repeated-token", "Lặp một từ nhiều lần, thiếu nội dung."

        # profanity-only / mostly profanity
        if (1 <= tok_cnt <= 6) and (prof_ratio >= 0.6) and pos_sig == 0 and neg_sig == 0:
            return True, "mostly-profanity", "Phần lớn từ ngữ là tục tĩu, thiếu nội dung đánh giá."

        # Câu toàn từ rất ngắn → chỉ khi rất ít từ
        if tok_cnt <= 3 and avg_len <= 3 and pos_sig == 0 and neg_sig == 0 and not any(self._is_product_hint_token(w) for w in toks_c):
            return True, "too-short-words", "Các từ quá ngắn, không đủ ngữ cảnh."

        # Letters ratio thấp mà không có tín hiệu nội dung
        if alpha_ratio < 0.40 and pos_sig == 0 and neg_sig == 0 and not any(self._is_product_hint_token(w) for w in toks_c):
            return True, "too-few-letters", "Tỷ lệ chữ cái quá thấp, thiếu nội dung."

        # Ký hiệu nhiều + ngắn
        if sym_ratio > 0.35 and len(t) < 30 and pos_sig == 0 and neg_sig == 0:
            return True, "too-many-symbols", "Ký hiệu/emoji quá nhiều, thiếu nội dung."

        # Đa dạng ký tự thấp + câu ngắn
        if uniq_ratio < 0.15 and len(t) < 20 and pos_sig == 0 and neg_sig == 0:
            return True, "low-variance", "Đa dạng ký tự rất thấp, nghi ngờ vô nghĩa."

        # Đặc biệt: 'bt/bth' đi cùng danh từ sản phẩm => giữ lại (neutral)
        if any(w in {"bt","bth"} for w in toks_c) and any(self._is_product_hint_token(w) for w in toks_c):
            return False, "", ""

        return False, "", ""

    def _is_single_token_noise(self, text: str) -> bool:
        toks = re.findall(r"\w+", text.strip().lower(), re.UNICODE)
        if len(toks) != 1: return False
        t0 = toks[0]
        if self._is_product_hint_token(t0):
            return False
        L = len(t0)
        nod = (tp.strip_accents_simple(t0) if tp else self._strip_diacritics_basic(t0))
        dia_ratio = (tp.approx_diacritic_ratio(t0) if tp else 0.0)
        cons_ratio = self._consonant_ratio(nod); vow_ratio = self._vowel_ratio(nod)
        letters = sum(ch.isalpha() for ch in t0); digits = sum(ch.isdigit() for ch in t0)
        if digits > 0 and letters < 3 and L >= 5:
            return True
        if L >= 12: return True
        if 6 <= L <= 11:
            if self._has_repeating_chunk(nod): return True
            if cons_ratio >= 0.75: return True
            if dia_ratio < 0.05: return True
        if 3 <= L <= 5 and (vow_ratio <= 0.25 or re.search(r"[^aeiouy\d]{3,}", nod)):
            return True
        # Thêm: 1–2 ký tự không có nguyên âm => rác
        if 1 <= L <= 2 and not self._has_vi_vowel(t0):
            return True
        return False

    # ---------- Engine toggle ----------
    def _on_engine_toggled(self, checked: bool):
        self.use_llm = checked
        if checked:
            self.btn_engine.setText("Engine: LLM (HTTP)")
            self.btn_engine.setToolTip("Đang dùng LLM (HTTP). Bấm để chuyển về PhoBERT.")
            self.llm_opts.show()
        else:
            self.btn_engine.setText("Engine: PhoBERT")
            self.btn_engine.setToolTip("Đang dùng PhoBERT. Bấm để chuyển sang LLM (HTTP).")
            self.llm_opts.hide()

    # ---------- LLM HTTP classify ----------
    # ---------- LLM HTTP classify ----------
    def _classify_llm_runner(self, url: str, model: str) -> Callable[[str], str]:
        LABELS = {"very_positive", "positive", "neutral", "negative", "very_negative"}

        def _normalize_str(s: str) -> str:
            s = unicodedata.normalize("NFKC", s)
            return s.strip().lower()

        # alias để bắt mấy kiểu viết khác / tiếng Việt
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

        def _extract_label(raw: str) -> str:
            """Cố lấy ra 1 nhãn trong LABELS từ chuỗi model trả về."""
            if not raw:
                return "neutral"
            s = _normalize_str(raw)

            # 1) nếu trong string có trực tiếp nhãn chuẩn
            for lab in LABELS:
                if lab in s:
                    return lab

            # 2) nếu match alias (tiếng Việt / viết thường)
            for key, lab in alias_map.items():
                if key in s:
                    return lab

            # 3) tách token chữ cái / underscore, thử từng token
            tokens = re.findall(r"[a-z_]+", s)
            for tok in tokens:
                t = _normalize_str(tok).replace("-", "_").strip("._ ")
                if t in alias_map:
                    return alias_map[t]
                if t in LABELS:
                    return t

            # bó tay thì cho neutral
            return "neutral"

        SYSTEM_PROMPT = (
            "Bạn là bộ phân loại cảm xúc bình luận sản phẩm tiếng Việt.\n"
            "Nhãn hợp lệ: very_positive, positive, neutral, negative, very_negative.\n\n"
            "Nhiệm vụ:\n"
            "- Đọc bình luận sản phẩm.\n"
            "- Chọn MỘT nhãn duy nhất thể hiện cảm xúc tổng thể.\n"
            "- CHỈ trả về đúng MỘT từ trong các từ sau (không thêm bất kỳ chữ nào khác):\n"
            "very_positive\npositive\nneutral\nnegative\nvery_negative\n"
        )

        def _runner(text: str) -> str:
            if requests is None:
                raise RuntimeError("Thiếu thư viện 'requests'. Hãy cài: pip install requests")

            prompt = f"{SYSTEM_PROMPT}\n\nBình luận: \"{text}\"\nNhãn:"

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                # 👇 infer xong là unload model khỏi RAM/GPU
                "keep_alive": 0,
                "options": {
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "num_ctx": 2048,
                },
            }

            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            out = r.json().get("response", "").strip()

            label = _extract_label(out)
            return f"LLM label: {label}"

        return _runner


    # ---------- Clear chat ----------
    def on_clear_chat(self):
        self.history.clear(); self.input.clear()
        self.status.setText("Cleared"); self.status.setStyleSheet("color: gray;")
        self.append_info("Đã xoá toàn bộ hội thoại.")

    # ---------- Events ----------
    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QtCore.QEvent.Type.KeyPress:
            key_event: QtGui.QKeyEvent = event  # type: ignore[assignment]
            if key_event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if key_event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    return False
                self.on_send(); return True
        return super().eventFilter(obj, event)

    # ---------- Slots ----------
    def on_model_loaded(self, name: str):
        self.lbl_model.setText(f"Loaded: {name}"); self.lbl_model.setStyleSheet("color: #2e7d32;")
        if self.model.has_normalize():
            self.append_info("Module có normalize(). Bạn có thể bật 'Use normalizer' (nếu cần).")
        else:
            self.append_info("Module không có normalize(). Sẽ chỉ dùng infer().")

    def on_model_cleared(self):
        self.lbl_model.setText("No model loaded"); self.lbl_model.setStyleSheet("color: gray;")

    def on_send(self):
        text = self.input.toPlainText().strip()
        if not text: return
        self.append_user(text); self.input.clear()

        is_bad, _, reason_msg = self._is_low_info(text)
        if is_bad:
            skipped = (
                f"Text: {text}\n"
                f"Pred: <skipped>\n"
                f"Note: Bỏ qua đầu vào — {reason_msg}\n"
                f"Hint: Hãy nhập câu có nội dung đánh giá sản phẩm (vd: 'Chất lượng ổn, bass hơi yếu')."
            )
            self.append_model(skipped)
            self.status.setText("Ready"); self.status.setStyleSheet("color: gray;")
            return

        self.status.setText("Running…"); self.status.setStyleSheet("color: #1565c0;")

        if self.use_llm:
            url = self.txt_llm_url.text().strip()
            model = self.cmb_llm_model.currentText().strip()
            runner = self._classify_llm_runner(url=url, model=model)
        else:
            runner = lambda t: self.model.infer(t, use_normalize=False)

        th = QtCore.QThread(self)
        worker = InferWorker(runner, text)
        worker.moveToThread(th)
        th.started.connect(worker.run)
        worker.finished.connect(self.on_infer_finished)
        worker.failed.connect(self.on_infer_failed)
        worker.finished.connect(lambda *_: self._finish_thread(th, worker))
        worker.failed.connect(lambda *_: self._finish_thread(th, worker))
        th.start(); self._active_threads.append(th)

    def _finish_thread(self, th: QtCore.QThread, worker: InferWorker):
        th.quit(); th.wait(2000)
        try: self._active_threads.remove(th)
        except ValueError: pass
        worker.deleteLater(); th.deleteLater()

    @QtCore.pyqtSlot(str, str)
    def on_infer_finished(self, input_text: str, output_text: str):
        self.append_model(output_text)
        self.status.setText("Ready"); self.status.setStyleSheet("color: gray;")

    @QtCore.pyqtSlot(str)
    def on_infer_failed(self, err: str):
        self.append_error(err)
        self.status.setText("Failed"); self.status.setStyleSheet("color: #c62828;")

    def on_save_log(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Lưu hội thoại", "chat_log.txt", "Text Files (*.txt)")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.history.toPlainText())
        except Exception as e:
            self.append_error(f"Không thể lưu: {e}")
        else:
            self.append_info(f"Đã lưu: {path}")

def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    if cwd not in sys.path: sys.path.insert(0, cwd)
    app = QtWidgets.QApplication(sys.argv)
    win = ChatWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
