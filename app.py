import os
import time
import sys
import cv2
import mss
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI
from dotenv import load_dotenv

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QLineEdit
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


# ---------- 监测线程：只识别，不自动调 AI ----------
class MonitorThread(QThread):
    new_question = pyqtSignal(str)   # 识别到的新题目文字
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

                self.new_question.emit(text)
                self.status.emit(f"[{time.strftime('%H:%M:%S')}] 已识别到题目，写好要求后点『生成答案』")

                time.sleep(INTERVAL)

    def _ocr(self, frame):
        result, _ = ocr(frame)
        if not result:
            return ""
        return "\n".join(line[1] for line in result)

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.last_sig = None


# ---------- AI 生成线程：点按钮后才跑 ----------
class GenerateThread(QThread):
    new_answer = pyqtSignal(str)

    def __init__(self, question, hint):
        super().__init__()
        self.question = question
        self.hint = hint

    def run(self):
        try:
            sys_prompt = "你是一个答题助手。用户会给你屏幕上的题目文字，可能包含多道题。请针对每一道能识别出的题目，直接给出答案和简要过程。"
            if self.hint.strip():
                sys_prompt += f"\n\n用户的额外要求：{self.hint.strip()}"

            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": self.question},
                ],
            )
            self.new_answer.emit(resp.choices[0].message.content)
        except Exception as e:
            self.new_answer.emit(f"[错误] {e}")


# ---------- 浮窗 ----------
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(840, 800)
        self.move(1060, 60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("答题助手")
        title.setStyleSheet("color: #8ecaff; font-size: 18px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        self.status_label = QLabel("未开始")
        self.status_label.setStyleSheet("color: #bbbbbb; font-size: 13px; background: transparent;")
        layout.addWidget(self.status_label)

        # ---- 给 AI 的要求输入框 ----
        hint_label = QLabel("给 AI 的要求（可选，改完点『生成答案』）")
        hint_label.setStyleSheet("color: #cccccc; font-size: 13px; background: transparent;")
        layout.addWidget(hint_label)

        self.hint_input = QLineEdit()
        self.hint_input.setPlaceholderText("例如：用 Python 写，只给代码")
        self.hint_input.setStyleSheet("""
            QLineEdit {
                background: rgba(40, 40, 55, 220);
                color: #ffffff;
                border: 1px solid rgba(120,120,160,150);
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.hint_input)

        # ---- 答案滚动区 ----
        ans_label = QLabel("答案")
        ans_label.setStyleSheet("color: #cccccc; font-size: 13px; background: transparent;")
        layout.addWidget(ans_label)

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

        self.gen_btn = QPushButton("生成答案")
        self.gen_btn.clicked.connect(self.generate)
        self.gen_btn.setEnabled(False)   # 没识别到题目时不能点
        btn_row.addWidget(self.gen_btn)

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
            QPushButton:disabled {
                background: rgba(60, 60, 75, 180);
                color: rgba(180,180,180,120);
            }
        """
        self.toggle_btn.setStyleSheet(btn_style)
        self.gen_btn.setStyleSheet(btn_style)
        close_btn.setStyleSheet(btn_style)

        self._drag_pos = None
        self.monitor_thread = None
        self.gen_thread = None
        self.current_question = ""   # 最近一次识别到的题目

    # ---------- 画不透明背景 ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(15, 15, 20, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(self.rect()), 12, 12)

    # ---------- 检测控制 ----------
    def toggle_monitor(self):
        if self.monitor_thread is None or not self.monitor_thread.isRunning():
            self.start_monitor()
        else:
            self.stop_monitor()

    def start_monitor(self):
        self.monitor_thread = MonitorThread()
        self.monitor_thread.new_question.connect(self.on_new_question)
        self.monitor_thread.status.connect(self.set_status)
        self.monitor_thread.running = True
        self.monitor_thread.reset()
        self.monitor_thread.start()
        self.toggle_btn.setText("停止检测")
        self.set_status("监测中，请把题目放在左半屏...")

    def stop_monitor(self):
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread = None
        self.toggle_btn.setText("开始检测")
        self.set_status("已停止")

    # ---------- 收到新题目 ----------
    def on_new_question(self, text):
        self.current_question = text
        self.gen_btn.setEnabled(True)
        self.content.setText("（已识别到题目，写好要求后点『生成答案』）")

    # ---------- 生成答案 ----------
    def generate(self):
        if not self.current_question:
            self.set_status("还没识别到题目")
            return
        hint = self.hint_input.text()
        self.set_status("AI 思考中...")
        self.gen_btn.setEnabled(False)
        self.content.setText("正在生成答案...")

        self.gen_thread = GenerateThread(self.current_question, hint)
        self.gen_thread.new_answer.connect(self.on_answer)
        self.gen_thread.start()

    def on_answer(self, text):
        self.content.setText(text)
        self.set_status(f"[{time.strftime('%H:%M:%S')}] 完成")
        self.gen_btn.setEnabled(True)

    # ---------- 退出 ----------
    def exit_app(self):
        if self.monitor_thread:
            self.monitor_thread.stop()
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


# ---------- 主程序 ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Overlay()
    win.show()
    sys.exit(app.exec_())