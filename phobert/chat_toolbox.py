#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Toolbox GUI (PyQt6) — Auto-load my_phobert_only.py, toggle PhoBERT <-> LLM (HTTP)
V1.8 — Gate vô nghĩa:
  • Tích hợp textproc (normalize/count_lexicon/diacritic)
  • Fallback cụm ngắn tích cực (xịn xò/xịn sò/quá xịn/...)
  • Tautology 'X như|là|thì X' & Generic-only
  • NEW: Generic + Gibberish (vd: 'sản phâm sfasjf') -> skip
"""
from __future__ import annotations

import sys, os, traceback, importlib, importlib.util, html, json, re, unicodedata
from typing import Callable, Optional
from collections import Counter
from PyQt6 import QtCore, QtWidgets, QtGui

# ====== import textproc (file bạn tự code) ======
try:
    import textproc as tp
except Exception:
    tp = None  # vẫn chạy được, chỉ mất 1 số tín hiệu

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

    _PRODUCT_HINTS = re.compile(
        r"(sản|phẩm|hàng|shop|đơn|đóng|gói|màn|hình|loa|tai|nghe|âm|bass|pin|sạc|"
        r"chuột|bàn|phím|điện|thoại|ốp|miếng|dán|áo|quần|giày|dép|đẹp|tốt|xấu|ổn|ok|oke)",
        re.UNICODE
    )

    _CONSONANTS = set(chr(c) for c in range(97,123)) - set("aeiouy"); _CONSONANTS.update(list("đ"))

    # Fallback cụm tích cực ngắn (cứu khi textproc không match)
    _LOCAL_POS_FALLBACK = [
        re.compile(r"\bxịn\s*[sx][òo]\b", re.IGNORECASE),  # xịn sò / xịn xò
        re.compile(r"\bquá\s*xịn\b", re.IGNORECASE),
        re.compile(r"\bquá\s*ok(?:e+)?\b", re.IGNORECASE),
        re.compile(r"\bshop\s*(ổn|ok|oke|uy\s*tín|nhiệt\s*tình)\b", re.IGNORECASE),
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
        self.cmb_llm_model.addItems(["qwen2.5:14b-instruct","qwen2.5:7b-instruct","llama3.1:8b-instruct","gemma2:9b-instruct","phi3:mini"])
        self.cmb_llm_model.setCurrentText("qwen2.5:14b-instruct"); self.cmb_llm_model.setMinimumWidth(180)
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
    def _has_repeating_chunk(self, s: str) -> bool:
        if len(s) < 6: return False
        for k in (2, 3, 4):
            m = re.search(rf"([a-z0-9]{{{k}}})\1+", s)
            if m: return True
        return False
    def _norm_words(self, text: str):
        nod = (tp.strip_accents_simple(text) if tp else self._strip_diacritics_basic(text))
        return re.findall(r"[a-z0-9]+", nod.lower())

    # NEW: detect tautology & generic-only
    def _is_tautology_like(self, text: str) -> bool:
        nod = (tp.strip_accents_simple(text) if tp else self._strip_diacritics_basic(text)).lower()
        nod = re.sub(r"\s+", " ", nod).strip()
        if re.search(r"\b([a-z0-9]{2,}(?:\s+[a-z0-9]{2,}){0,3})\s+(nhu|la|thi)\s+\1\b", nod):
            return True
        toks = re.findall(r"[a-z0-9]+", nod)
        if toks:
            if all(t in self._GENERIC_TOKENS or t in self._FUNC_WORDS for t in toks):
                return True
        return False

    # NEW: generic + gibberish (vd: 'san pham sfasjf')
    def _is_generic_plus_gibberish(self, text: str) -> bool:
        toks = self._norm_words(text)
        if not toks: return False
        # tách generic/func và phần còn lại
        generic_or_func = [t for t in toks if (t in self._GENERIC_TOKENS or t in self._FUNC_WORDS)]
        others = [t for t in toks if t not in self._GENERIC_TOKENS and t not in self._FUNC_WORDS]
        if not others:
            return False
        # Nếu phần "khác" chỉ toàn 1–2 token và có >=1 token nghi rác -> true
        def _is_gib(t: str) -> bool:
            if any(ch.isdigit() for ch in t):  # có số thì coi là thông tin -> không xem là rác ở rule này
                return False
            L = len(t)
            nod = t
            cons_ratio = self._consonant_ratio(nod)
            if L >= 5 and (cons_ratio >= 0.75 or self._has_repeating_chunk(nod)):
                return True
            # rất ít dấu tiếng Việt (sau strip còn y như cũ) + không nằm trong product hints
            if L >= 5 and not self._PRODUCT_HINTS.search(t) and self._has_repeating_chunk(nod):
                return True
            return False
        gib_cnt = sum(_is_gib(x) for x in others)
        # Chỉ kích hoạt khi đa số phần còn lại là gibberish và phần generic chiếm đa số tổng token
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

    def _is_single_token_noise(self, text: str) -> bool:
        toks = re.findall(r"\w+", text.strip().lower(), re.UNICODE)
        if len(toks) != 1: return False
        t0 = toks[0]
        if re.search(r"\d", t0): return False
        if self._PRODUCT_HINTS.search(t0): return False
        L = len(t0)
        nod = (tp.strip_accents_simple(t0) if tp else self._strip_diacritics_basic(t0))
        dia_ratio = (tp.approx_diacritic_ratio(t0) if tp else 0.0)
        cons_ratio = self._consonant_ratio(nod)
        if L >= 12: return True
        if 6 <= L <= 11:
            if self._has_repeating_chunk(nod): return True
            if cons_ratio >= 0.75: return True
            if dia_ratio < 0.05: return True
        return False

    def _is_low_info(self, text: str):
        """
        Trả về (True/False, reason_key, reason_msg).
        Dùng snapshot đã normalize của textproc để quyết định, nhưng KHÔNG sửa input gửi vào model.
        """
        t = text.strip()
        if t == "": return True, "empty", "Văn bản trống."
        if self._PAT_ALO.match(t): return True, "greeting/test-mic", "Chuỗi chào/kiểm tra micro, không đủ ngữ cảnh."
        if self._PAT_REPEAT_SYL.match(t.replace(".", "").replace("!", "").lower()):
            return True, "repeated-syllables", "Chuỗi lặp âm tiết, không chứa ý kiến về sản phẩm."
        if self._is_single_token_noise(t):
            return True, "single-token-gibberish", "Văn bản vô nghĩa: 1 từ không có nội dung đánh giá."
        if self._is_tautology_like(t):
            return True, "tautology", "Câu lặp lại cùng một cụm (ví dụ: 'X như X'), không chứa nhận xét."
        if self._is_generic_plus_gibberish(t):
            return True, "generic+gibberish", "Chỉ chứa từ chung + 1–2 từ vô nghĩa, không có nhận xét."

        alpha_ratio, uniq_ratio, sym_ratio, tok_cnt, bad_tok_cnt, toks_c, toks, avg_len, prof_ratio = self._sig_stats(t)

        # textproc lexicon signals
        t_norm = tp.normalize_text(t) if tp else t
        try:
            pos_sig, neg_sig = tp.count_lexicon(t_norm) if tp else (0, 0)
        except Exception:
            pos_sig, neg_sig = (0, 0)

        # LOCAL fallback: cứu cụm ngắn tích cực
        if pos_sig == 0 and neg_sig == 0:
            for pat in self._LOCAL_POS_FALLBACK:
                if pat.search(t):
                    pos_sig = 1
                    break

        # Repeated token ≥3 lần
        if tok_cnt >= 3:
            tok_most, n_most = Counter(toks_c).most_common(1)[0]
            if n_most >= 3 and (len(tok_most) <= 6 or self._token_is_profanity(tok_most)):
                if pos_sig == 0 and neg_sig == 0:
                    return True, "repeated-token", "Lặp một từ nhiều lần, thiếu nội dung."

        # profanity-only / mostly profanity
        if (1 <= tok_cnt <= 6) and (prof_ratio >= 0.6):
            if pos_sig == 0 and neg_sig == 0:
                return True, "mostly-profanity", "Phần lớn từ ngữ là tục tĩu, thiếu nội dung đánh giá."

        # Câu toàn từ rất ngắn
        if tok_cnt <= 4 and avg_len <= 3 and pos_sig == 0 and neg_sig == 0:
            return True, "too-short-words", "Các từ quá ngắn, không đủ ngữ cảnh."

        # Short/letters/symbols
        if tok_cnt <= 1 and len(t) < 6 and pos_sig == 0 and neg_sig == 0:
            return True, "too-short", "Quá ngắn, không đủ ngữ cảnh."
        if alpha_ratio < 0.25 and pos_sig == 0 and neg_sig == 0:
            return True, "too-few-letters", "Tỷ lệ chữ cái quá thấp."
        if sym_ratio > 0.35 and len(t) < 30 and pos_sig == 0 and neg_sig == 0:
            return True, "too-many-symbols", "Ký hiệu/emoji quá nhiều, thiếu nội dung."

        # Đa dạng ký tự thấp + câu ngắn
        if uniq_ratio < 0.15 and len(t) < 20 and pos_sig == 0 and neg_sig == 0:
            return True, "low-variance", "Đa dạng ký tự rất thấp, nghi ngờ vô nghĩa."

        return False, "", ""

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
    def _classify_llm_runner(self, url: str, model: str) -> Callable[[str], str]:
        LABELS = {"very_positive","positive","neutral","negative","very_negative"}
        SYSTEM_PROMPT = (
            "Bạn là bộ phân loại cảm xúc tiếng Việt.\n"
            "Chỉ trả lời đúng MỘT JSON duy nhất dạng: {\"label\":\"<một trong 5 nhãn>\"}\n"
            "Năm nhãn hợp lệ: very_positive, positive, neutral, negative, very_negative.\n"
            "Không giải thích, không thêm chữ nào ngoài JSON."
        )
        def _runner(text: str) -> str:
            if requests is None:
                raise RuntimeError("Thiếu thư viện 'requests'. Hãy cài: pip install requests")
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nPhân loại câu sau:\n\"{text}\"\nJSON:"
            payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": 0,
                       "options": {"temperature": 0, "top_p": 1, "seed": 42, "num_ctx": 2048}}
            r = requests.post(url, json=payload, timeout=120); r.raise_for_status()
            out = r.json().get("response", "").strip()
            m = re.search(r"\{.*\}", out, flags=re.DOTALL)
            label = "neutral"
            if m:
                try:
                    cand = str(json.loads(m.group(0)).get("label","")).strip().lower().replace(" ","_")
                    if cand in LABELS: label = cand
                except Exception:
                    label = "neutral"
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

        # Gate trước khi chạy model
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

        # Decide runner
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
