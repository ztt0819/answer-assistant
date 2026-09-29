import os
import time
import sys
import cv2
import mss
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI
from dotenv import load_dotenv

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea
)

load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("请先设置 DEEPSEEK_API_KEY 环境变量")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

REGION = {"left": 0, "top": 140, "width": 960, "height": 900}
INTERVAL = 1.5
CHANGE_THRESHOLD = 8

ocr = RapidOCR()


# ---------- 监测线程 ----------
class MonitorThread(QThread):
    new_answer = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.last_sig = None

    def run(self):
        with mss.MSS() as sct:
            while self.running:
                frame = np.array(sct.grab(REGION))[:, :, :3]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (32, 32)).astype(np.float32)

                if self.last_sig is not None:
                    diff = np.abs(small - self.last_sig).mean()
                    if diff < CHANGE_THRESHOLD:
                        time.sleep(INTERVAL)
                        continue

                self.last_sig = small
                self.status.emit(f"[{time.strftime('%H:%M:%S')}] 检测到画面变化，识别中...")

                text = self._ocr(frame)
                if not text.strip():
                    self.status.emit("未识别到文字，继续监测...")
                    time.sleep(INTERVAL)
                    continue

                self.status.emit(f"[{time.strftime('%H:%M:%S')}] 识别到 {len(text)} 字符，AI 思考中...")
                answer = self._ask(text)
                self.new_answer.emit(answer)
                self.status.emit(f"[{time.strftime('%H:%M:%S')}] 完成")

                time.sleep(INTERVAL)

    def _ocr(self, frame):
        result, _ = ocr(frame)
        if not result:
            return ""
        return "\n".join(line[1] for line in result)

    def _ask(self, text):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content":
                        "你是一个答题助手。用户会给你屏幕上的题目文字，可能包含多道题。"
                        "请针对每一道能识别出的题目，直接给出答案和简要过程。"
                        "用简洁的格式，不要啰嗦。"},
                    {"role": "user", "content": text},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[错误] {e}"

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.last_sig = None


# ---------- 浮窗 ----------
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import QRectF

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(840, 700)
        self.move(1060, 100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("答题助手")
        title.setStyleSheet("color: #8ecaff; font-size: 18px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        self.status_label = QLabel("未开始")
        self.status_label.setStyleSheet("color: #bbbbbb; font-size: 13px; background: transparent;")
        layout.addWidget(self.status_label)

        # ---- 滚动区 ----
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical {
                background: rgba(60,60,80,120); width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(140,140,180,180); border-radius: 4px;
            }
        """)

        self.content = QLabel("点『开始检测』后，将题目页放到屏幕左半边。")
        self.content.setWordWrap(True)
        self.content.setAlignment(Qt.AlignTop)
        self.content.setStyleSheet(
            "color: #ffe45c; font-size: 16px; font-weight: bold; background: transparent;"
        )
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)

        # ---- 按钮行 ----
        btn_row = QHBoxLayout()

        self.toggle_btn = QPushButton("开始检测")
        self.toggle_btn.clicked.connect(self.toggle_monitor)
        btn_row.addWidget(self.toggle_btn)

        close_btn = QPushButton("退出")
        close_btn.clicked.connect(self.exit_app)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

        btn_style = """
            QPushButton {
                background: rgba(90, 90, 120, 240);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-size: 15px;
            }
            QPushButton:hover {
                background: rgba(120, 120, 160, 255);
            }
        """
        self.toggle_btn.setStyleSheet(btn_style)
        close_btn.setStyleSheet(btn_style)

        self._drag_pos = None
        self.thread = None

    # ---------- 画不透明背景 ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(15, 15, 20, 255))   # 完全不透明
        painter.setPen(Qt.NoPen)
        rect = QRectF(self.rect())
        painter.drawRoundedRect(rect, 12, 12)        # 圆角 12

    # ---------- 监测控制 ----------
    def toggle_monitor(self):
        if self.thread is None or not self.thread.isRunning():
            self.start_monitor()
        else:
            self.stop_monitor()

    def start_monitor(self):
        self.thread = MonitorThread()
        self.thread.new_answer.connect(self.set_answer)
        self.thread.status.connect(self.set_status)
        self.thread.running = True
        self.thread.reset()
        self.thread.start()
        self.toggle_btn.setText("停止检测")
        self.set_status("监测中，请把题目放在左半屏...")

    def stop_monitor(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.toggle_btn.setText("开始检测")
        self.set_status("已停止")

    def exit_app(self):
        if self.thread:
            self.thread.stop()
        QApplication.quit()

    # ---------- 拖动 ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_pos and e.buttons() == Qt.LeftButton:
            self.move(e.globalPos() - self._drag_pos)
            e.accept()

    # ---------- 更新 ----------
    def set_status(self, text):
        self.status_label.setText(text)

    def set_answer(self, text):
        self.content.setText(text)

# ---------- 主程序 ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Overlay()
    win.show()
    sys.exit(app.exec_())