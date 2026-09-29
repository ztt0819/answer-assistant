import os
import time
import cv2
import mss
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI
from dotenv import load_dotenv          # ← 新增第 1 处

# 从 .env 文件加载环境变量
load_dotenv()                            # ← 新增第 2 处

# ---------- 配置 ----------
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("请先设置 DEEPSEEK_API_KEY 环境变量")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 只截左半屏（题目区）
REGION = {"left": 0, "top": 140, "width": 960, "height": 900}

# 检测间隔（秒）
INTERVAL = 1.5
# 画面变化阈值：低于这个值认为没变，跳过识别
CHANGE_THRESHOLD = 8

ocr = RapidOCR()


def capture():
    with mss.MSS() as sct:
        frame = np.array(sct.grab(REGION))
    return frame[:, :, :3]


def frame_signature(frame):
    """把画面压成一个小指纹，用来判断是否变化"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32))
    return small.astype(np.float32)


def ocr_text(frame):
    result, _ = ocr(frame)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def ask_deepseek(text):
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


def main():
    print("实时监测已启动，按 Ctrl+C 停止。")
    print(f"监测区域：{REGION}")
    print(f"间隔：{INTERVAL} 秒\n")

    last_sig = None
    last_answer = ""

    while True:
        try:
            frame = capture()
            sig = frame_signature(frame)

            # 判断画面是否变化
            if last_sig is not None:
                diff = np.abs(sig - last_sig).mean()
                if diff < CHANGE_THRESHOLD:
                    time.sleep(INTERVAL)
                    continue

            last_sig = sig
            print("\n" + "=" * 50)
            print(f"[{time.strftime('%H:%M:%S')}] 检测到画面变化，开始识别...")

            text = ocr_text(frame)
            if not text.strip():
                print("未识别到文字，跳过")
                time.sleep(INTERVAL)
                continue

            print(f"识别到 {len(text)} 个字符，调用 AI...")
            answer = ask_deepseek(text)
            last_answer = answer

            print("\n----- AI 答案 -----")
            print(answer)
            print("-" * 50)

            time.sleep(INTERVAL)

        except KeyboardInterrupt:
            print("\n\n已停止监测。")
            break
        except Exception as e:
            print(f"\n[错误] {e}")
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
    