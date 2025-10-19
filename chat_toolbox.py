#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Toolbox GUI (PyQt6)

Quick start:
    pip install PyQt6
    python3 chat_toolbox.py

Features:
    - Type text and press Enter to send (Shift+Enter for newline).
    - Non-blocking inference in a background thread.
    - Load a custom Python file that defines `infer(text: str) -> str`.
      Optionally, the same file may define `normalize(text: str) -> str`.
    - Toggle "Use normalizer" if your module provides normalize().
    - Save conversation as TXT.

Integrating your model:
    1) Create a small Python file, e.g. `my_model.py` with:
        def infer(text: str) -> str:
            # run your PhoBERT / MT5 / whatever and return a string
            return "predicted: " + text

        # (optional) if you have a normalizer
        def normalize(text: str) -> str:
            return text  # replace with your normalization

    2) In the GUI, click "Load Model Function..." and pick `my_model.py`.

Notes:
    - If you want to call code from a notebook, move the core logic into a plain .py helper
      and import that here, or have the helper file import from your package.
"""
from __future__ import annotations

import sys
import traceback
import importlib.util
from typing import Callable, Optional

from PyQt6 import QtCore, QtWidgets, QtGui


# -------------------------------
# Dynamic model loader
# -------------------------------
class ModelClient(QtCore.QObject):
    """Thin wrapper around a user-supplied infer() function (and optional normalize())."""
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
        spec.loader.exec_module(module)  # may raise

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
            # default echo if no model loaded
            return f"[echo] {text}"
        if use_normalize and self.has_normalize():
            try:
                text = self._normalize_fn(text)  # type: ignore[misc]
            except Exception:
                # Don't crash the app if normalize fails
                traceback.print_exc()
        return self._infer_fn(text)  # type: ignore[misc]


# -------------------------------
# Worker for background inference
# -------------------------------
class InferWorker(QtCore.QObject):
    started = QtCore.pyqtSignal(str)           # input text
    finished = QtCore.pyqtSignal(str, str)     # input text, output text
    failed = QtCore.pyqtSignal(str)            # error message

    def __init__(self, model: ModelClient, text: str, use_normalize: bool, parent=None):
        super().__init__(parent)
        self._model = model
        self._text = text
        self._use_normalize = use_normalize

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.started.emit(self._text)
            out = self._model.infer(self._text, use_normalize=self._use_normalize)
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
        self.resize(900, 600)

        self.model = ModelClient(self)

        # Central widget
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Top controls
        top_bar = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Model Function…")
        self.btn_clear_model = QtWidgets.QPushButton("Clear")
        self.chk_use_normalizer = QtWidgets.QCheckBox("Use normalizer() if available")
        self.lbl_model = QtWidgets.QLabel("No model loaded")
        self.lbl_model.setStyleSheet("color: gray;")
        top_bar.addWidget(self.btn_load)
        top_bar.addWidget(self.btn_clear_model)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.chk_use_normalizer)
        top_bar.addStretch(1)
        top_bar.addWidget(self.lbl_model)
        main_layout.addLayout(top_bar)

        # Chat history (read-only)
        self.history = QtWidgets.QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setPlaceholderText("Lịch sử hội thoại sẽ hiển thị ở đây…")
        self.history.setMinimumHeight(300)
        main_layout.addWidget(self.history, 1)

        # Input area + send
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

        # Bottom bar
        bottom_bar = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Log…")
        self.status = QtWidgets.QLabel("Ready")
        self.status.setStyleSheet("color: gray;")
        bottom_bar.addWidget(self.btn_save)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.status)
        main_layout.addLayout(bottom_bar)

        # Signals
        self.btn_load.clicked.connect(self.on_load_model)
        self.btn_clear_model.clicked.connect(self.on_clear_model)
        self.btn_send.clicked.connect(self.on_send)
        self.btn_save.clicked.connect(self.on_save_log)
        self.model.modelLoaded.connect(self.on_model_loaded)
        self.model.modelCleared.connect(self.on_model_cleared)

        # Thread bookkeeping
        self._active_threads: list[QtCore.QThread] = []

    # ---------- UI helpers ----------
    def append_user(self, text: str):
        self.history.appendPlainText(f"You: {text}")

    def append_model(self, text: str):
        self.history.appendPlainText(f"Model: {text}")

    def append_info(self, text: str):
        self.history.appendPlainText(f"[Info] {text}")

    def append_error(self, text: str):
        self.history.appendPlainText(f"[Error] {text}")

    # ---------- Events ----------
    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QtCore.QEvent.Type.KeyPress:
            key_event: QtGui.QKeyEvent = event  # type: ignore[assignment]
            if key_event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if key_event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    # allow newline with Shift+Enter
                    return False
                # send on Enter
                self.on_send()
                return True
        return super().eventFilter(obj, event)

    # ---------- Slots ----------
    def on_model_loaded(self, path: str):
        self.lbl_model.setText(f"Loaded: {path}")
        self.lbl_model.setStyleSheet("color: #2e7d32;")  # green
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
            self.status.setStyleSheet("color: #c62828;")  # red
        else:
            self.status.setText("Model loaded")
            self.status.setStyleSheet("color: #2e7d32;")  # green

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
        self.status.setStyleSheet("color: #1565c0;")  # blue

        # Spin up a worker thread
        th = QtCore.QThread(self)
        worker = InferWorker(self.model, text, self.chk_use_normalizer.isChecked())
        worker.moveToThread(th)
        th.started.connect(worker.run)
        worker.started.connect(lambda _: None)
        worker.finished.connect(self.on_infer_finished)
        worker.failed.connect(self.on_infer_failed)
        # Ensure cleanup
        worker.finished.connect(lambda *_: self._finish_thread(th, worker))
        worker.failed.connect(lambda *_: self._finish_thread(th, worker))
        th.start()
        self._active_threads.append(th)

    def _finish_thread(self, th: QtCore.QThread, worker: InferWorker):
        th.quit()
        th.wait(2000)
        # deref
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
        self.status.setStyleSheet("color: #c62828;")  # red

    def on_save_log(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Lưu hội thoại", "chat_log.txt", "Text Files (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
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
