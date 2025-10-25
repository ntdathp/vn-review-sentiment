#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Toolbox GUI (PyQt6) — Toolbar toggle PhoBERT <-> LLM (HTTP)
"""
from __future__ import annotations

import sys
import traceback
import importlib.util
import html
import json
import re
from typing import Callable, Optional

from PyQt6 import QtCore, QtWidgets, QtGui

# Optional: HTTP client for LLM
try:
    import requests
except Exception:
    requests = None  # chỉ báo lỗi khi bật LLM

# Colors
YOU_COLOR   = "#6a1b9a"  # purple
MODEL_COLOR = "#1565c0"  # blue
INFO_COLOR  = "#2e7d32"  # green
ERROR_COLOR = "#c62828"  # red
TEXT_COLOR  = "#000000"  # black


# -------------------------------
# Dynamic model loader (Python file with infer()/normalize())
# -------------------------------
class ModelClient(QtCore.QObject):
    modelLoaded = QtCore.pyqtSignal(str)  # path
    modelCleared = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._infer_fn: Optional[Callable[[str], str]] = None
        self._normalize_fn: Optional[Callable[[str], str]] = None
        self._module_path: Optional[str] = None

    def load_from_pyfile(self, path: str):
        spec = importlib.util.spec_from_file_location("chattoolbox_user_module", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Không thể load module từ đường dẫn đã chọn.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        infer_fn = getattr(module, "infer", None)
        if not callable(infer_fn):
            raise AttributeError("File đã chọn không định nghĩa hàm infer(text: str) -> str.")

        normalize_fn = getattr(module, "normalize", None)
        if normalize_fn is not None and not callable(normalize_fn):
            normalize_fn = None

        self._infer_fn = infer_fn
        self._normalize_fn = normalize_fn
        self._module_path = path
        self.modelLoaded.emit(path)

    def clear(self):
        self._infer_fn = None
        self._normalize_fn = None
        self._module_path = None
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
# Generic worker for background jobs
# -------------------------------
class InferWorker(QtCore.QObject):
    started = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, runner: Callable[[str], str], text: str, parent=None):
        super().__init__(parent)
        self._runner = runner
        self._text = text

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.started.emit(self._text)
            out = self._runner(self._text)
            if not isinstance(out, str):
                out = str(out)
            self.finished.emit(self._text, out)
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")


# -------------------------------
# Main Window
# -------------------------------
class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat Toolbox (PyQt6)")
        self.resize(1100, 680)
        self.setMinimumWidth(960)

        self.model = ModelClient(self)

        # ---- Engine state: mặc định PhoBERT (False = PhoBERT, True = LLM)
        self.use_llm = False

        # =================== TOOLBAR (luôn hiển thị)
        self.toolbar = QtWidgets.QToolBar("Engine")
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Toggle action (checkable)
        self.act_toggle = QtGui.QAction("Use LLM (HTTP)", self)
        self.act_toggle.setCheckable(True)  # checked = LLM, unchecked = PhoBERT
        self.act_toggle.setToolTip("Đang dùng PhoBERT (infer()). Bấm để chuyển sang LLM (HTTP).")
        self.toolbar.addAction(self.act_toggle)
        self.toolbar.addSeparator()

        # LLM model combo
        self.cmb_llm_model = QtWidgets.QComboBox()
        self.cmb_llm_model.setEditable(True)
        self.cmb_llm_model.addItems([
            "qwen2.5:14b-instruct",
            "qwen2.5:7b-instruct",
            "llama3.1:8b-instruct",
            "gemma2:9b-instruct",
            "phi3:mini"
        ])
        self.cmb_llm_model.setCurrentText("qwen2.5:14b-instruct")
        self.cmb_llm_model.setMinimumWidth(180)
        wa_model = QtWidgets.QWidgetAction(self)
        w_model = QtWidgets.QWidget()
        h1 = QtWidgets.QHBoxLayout(w_model)
        h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(QtWidgets.QLabel("Model:"))
        h1.addWidget(self.cmb_llm_model)
        wa_model.setDefaultWidget(w_model)
        self.toolbar.addAction(wa_model)

        # LLM URL line edit
        self.txt_llm_url = QtWidgets.QLineEdit("http://localhost:11434/api/generate")
        self.txt_llm_url.setMinimumWidth(260)
        self.txt_llm_url.setPlaceholderText("http://<host>:<port>/api/generate")
        wa_url = QtWidgets.QWidgetAction(self)
        w_url = QtWidgets.QWidget()
        h2 = QtWidgets.QHBoxLayout(w_url)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(QtWidgets.QLabel("URL:"))
        h2.addWidget(self.txt_llm_url)
        wa_url.setDefaultWidget(w_url)
        self.toolbar.addAction(wa_url)

        # Spacer to push label to the right
        self._toolbar_spacer = QtWidgets.QWidget()
        self._toolbar_spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(self._toolbar_spacer)

        # Model path label
        self.lbl_model = QtWidgets.QLabel("No model loaded")
        self.lbl_model.setStyleSheet("color: gray;")
        self.toolbar.addWidget(self.lbl_model)

        # Ban đầu (PhoBERT): ẩn options LLM
        w_model.setVisible(False)
        w_url.setVisible(False)
        self._w_model = w_model
        self._w_url = w_url

        # =================== Central widget
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # =================== Top controls (load/clear + normalizer)
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(8)

        self.btn_load = QtWidgets.QPushButton("Load Model Function…")
        self.btn_clear_model = QtWidgets.QPushButton("Clear")
        self.chk_use_normalizer = QtWidgets.QCheckBox("Use normalizer() if available")
        self.chk_use_normalizer.setToolTip("Bật nếu module .py có hàm normalize()")

        top_bar.addWidget(self.btn_load)
        top_bar.addWidget(self.btn_clear_model)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.chk_use_normalizer)
        top_bar.addStretch(1)
        main_layout.addLayout(top_bar)

        # =================== Chat history
        self.history = QtWidgets.QTextEdit()
        self.history.setReadOnly(True)
        self.history.setPlaceholderText("Lịch sử hội thoại sẽ hiển thị ở đây…")
        self.history.setMinimumHeight(320)
        main_layout.addWidget(self.history, 1)

        # =================== Input area + send
        input_row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText("Nhập tin nhắn… (Enter để gửi, Shift+Enter để xuống dòng)")
        self.input.installEventFilter(self)  # for Enter behavior
        self.btn_send = QtWidgets.QPushButton("Send")
        self.btn_send.setDefault(True)
        self.btn_send.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.btn_send)
        main_layout.addLayout(input_row)

        # =================== Bottom bar (Save | Clear Chat | Status)
        bottom_bar = QtWidgets.QHBoxLayout()
        bottom_bar.setSpacing(8)
        bottom_bar.setContentsMargins(0, 0, 0, 0)

        self.btn_save = QtWidgets.QPushButton("Save Log…")
        self.btn_clear_chat = QtWidgets.QPushButton("Clear Chat")
        bottom_bar.addWidget(self.btn_save)
        bottom_bar.addWidget(self.btn_clear_chat)
        bottom_bar.addStretch(1)

        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("color: gray;")
        self.status.setFixedWidth(90)
        bottom_bar.addWidget(self.status)

        main_layout.addLayout(bottom_bar)

        # Signals
        self.btn_load.clicked.connect(self.on_load_model)
        self.btn_clear_model.clicked.connect(self.on_clear_model)
        self.btn_send.clicked.connect(self.on_send)
        self.btn_save.clicked.connect(self.on_save_log)
        self.btn_clear_chat.clicked.connect(self.on_clear_chat)
        self.model.modelLoaded.connect(self.on_model_loaded)
        self.model.modelCleared.connect(self.on_model_cleared)
        self.act_toggle.toggled.connect(self._on_toggle_engine)

        # Threads
        self._active_threads: list[QtCore.QThread] = []

        # init engine UI state (mặc định PhoBERT)
        self.use_llm = False
        self._update_engine_ui()

        # In ra console cho chắc: PyQt version + state
        print("PyQt6 version:", QtCore.QT_VERSION_STR)
        print("Engine initial:", "PhoBERT")

    # ---------- UI helpers ----------
    def _append_block(self, title_html: str, body_text: str):
        safe_body = html.escape(body_text)
        block = (
            f"{title_html}<br>"
            f'<span style="color:{TEXT_COLOR}; white-space:pre-wrap;">{safe_body}</span><br>'
            f"<br>"
        )
        self.history.append(block)

    def append_user(self, text: str):
        title = f'<span style="color:{YOU_COLOR}; font-weight:600;">You:</span>'
        self._append_block(title, text)

    def append_model(self, text: str):
        title = f'<span style="color:{MODEL_COLOR}; font-weight:600;">Model:</span>'
        self._append_block(title, text)

    def append_info(self, text: str):
        title = f'<span style="color:{INFO_COLOR}; font-weight:600;">[Info]</span>'
        self._append_block(title, text)

    def append_error(self, text: str):
        title = f'<span style="color:{ERROR_COLOR}; font-weight:600;">[Error]</span>'
        self._append_block(title, text)

    # ---------- Engine toggle logic ----------
    def _on_toggle_engine(self, checked: bool):
        self.use_llm = checked  # True -> LLM, False -> PhoBERT
        self._update_engine_ui()

    def _update_engine_ui(self):
        if self.use_llm:
            self.act_toggle.setText("Use PhoBERT")
            self.act_toggle.setToolTip("Đang dùng LLM (HTTP). Bấm để chuyển về PhoBERT.")
            self._w_model.show()
            self._w_url.show()
        else:
            self.act_toggle.setText("Use LLM (HTTP)")
            self.act_toggle.setToolTip("Đang dùng PhoBERT (infer()). Bấm để chuyển sang LLM (HTTP).")
            self._w_model.hide()
            self._w_url.hide()

    # ---------- LLM HTTP classify ----------
    def _classify_llm_runner(self, url: str, model: str) -> Callable[[str], str]:
        LABELS = {"very_positive", "positive", "neutral", "negative", "very_negative"}
        SYSTEM_PROMPT = (
            "Bạn là bộ phân loại cảm xúc tiếng Việt.\n"
            "Chỉ trả lời đúng MỘT JSON duy nhất dạng: {\"label\":\"<một trong 5 nhãn>\"}\n"
            "Năm nhãn hợp lệ: very_positive, positive, neutral, negative, very_negative.\n"
            "Không giải thích, không thêm chữ nào ngoài JSON."
        )

        def _runner(text: str) -> str:
            if requests is None:
                raise RuntimeError("Thiếu thư viện 'requests'. Hãy cài: pip install requests")
            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\nPhân loại câu sau:\n\"{text}\"\nJSON:"
            )
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,
                "options": {"temperature": 0, "top_p": 1, "seed": 42, "num_ctx": 2048}
            }
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            out = r.json().get("response", "").strip()
            m = re.search(r"\{.*\}", out, flags=re.DOTALL)
            label = "neutral"
            if m:
                try:
                    obj = json.loads(m.group(0))
                    cand = str(obj.get("label", "")).strip().lower().replace(" ", "_")
                    if cand in LABELS:
                        label = cand
                except Exception:
                    label = "neutral"
            return f"LLM label: {label}"
        return _runner

    # ---------- Clear chat ----------
    def on_clear_chat(self):
        self.history.clear()
        self.input.clear()
        self.status.setText("Cleared")
        self.status.setStyleSheet("color: gray;")
        self.append_info("Đã xoá toàn bộ hội thoại.")

    # ---------- Events ----------
    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QtCore.QEvent.Type.KeyPress:
            key_event: QtGui.QKeyEvent = event  # type: ignore[assignment]
            if key_event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if key_event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    return False
                self.on_send()
                return True
        return super().eventFilter(obj, event)

    # ---------- Slots ----------
    def on_model_loaded(self, path: str):
        self.lbl_model.setText(f"Loaded: {path}")
        self.lbl_model.setStyleSheet("color: #2e7d32;")
        if self.model.has_normalize():
            self.chk_use_normalizer.setEnabled(True)
            self.append_info("Module có normalize(). Bạn có thể bật 'Use normalizer'.")
        else:
            self.chk_use_normalizer.setChecked(False)
            self.chk_use_normalizer.setEnabled(False)
            self.append_info("Module không có normalize(). Sẽ chỉ dùng infer().")

    def on_model_cleared(self):
        self.lbl_model.setText("No model loaded")
        self.lbl_model.setStyleSheet("color: gray;")
        self.chk_use_normalizer.setChecked(False)
        self.chk_use_normalizer.setEnabled(False)

    def on_load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn file .py chứa hàm infer()", "", "Python Files (*.py)"
        )
        if not path:
            return
        try:
            self.model.load_from_pyfile(path)
        except Exception as e:
            self.append_error(str(e))
            self.status.setText("Load failed")
            self.status.setStyleSheet("color: #c62828;")
        else:
            self.status.setText("Model loaded")
            self.status.setStyleSheet("color: #2e7d32;")

    def on_clear_model(self):
        self.model.clear()
        self.append_info("Đã xóa model hiện tại.")
        self.status.setText("Cleared")
        self.status.setStyleSheet("color: gray;")

    def on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.append_user(text)
        self.input.clear()
        self.status.setText("Running…")
        self.status.setStyleSheet("color: #1565c0;")

        # Decide runner: LLM HTTP vs Python infer()
        if self.use_llm:
            url = self.txt_llm_url.text().strip()
            model = self.cmb_llm_model.currentText().strip()
            runner = self._classify_llm_runner(url=url, model=model)
        else:
            use_norm = self.chk_use_normalizer.isChecked()
            runner = lambda t: self.model.infer(t, use_normalize=use_norm)

        th = QtCore.QThread(self)
        worker = InferWorker(runner, text)
        worker.moveToThread(th)
        th.started.connect(worker.run)
        worker.finished.connect(self.on_infer_finished)
        worker.failed.connect(self.on_infer_failed)
        worker.finished.connect(lambda *_: self._finish_thread(th, worker))
        worker.failed.connect(lambda *_: self._finish_thread(th, worker))
        th.start()
        self._active_threads.append(th)

    def _finish_thread(self, th: QtCore.QThread, worker: InferWorker):
        th.quit()
        th.wait(2000)
        try:
            self._active_threads.remove(th)
        except ValueError:
            pass
        worker.deleteLater()
        th.deleteLater()

    @QtCore.pyqtSlot(str, str)
    def on_infer_finished(self, input_text: str, output_text: str):
        self.append_model(output_text)
        self.status.setText("Ready")
        self.status.setStyleSheet("color: gray;")

    @QtCore.pyqtSlot(str)
    def on_infer_failed(self, err: str):
        self.append_error(err)
        self.status.setText("Failed")
        self.status.setStyleSheet("color: #c62828;")

    def on_save_log(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu hội thoại", "chat_log.txt", "Text Files (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f):
                f.write(self.history.toPlainText())
        except Exception as e:
            self.append_error(f"Không thể lưu: {e}")
        else:
            self.append_info(f"Đã lưu: {path}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
