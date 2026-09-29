# 答题助手

实时截屏识别题目，调用 DeepSeek AI 给出答案的 Windows 桌面工具。

## 功能

- 实时监测屏幕左半区的题目变化
- RapidOCR 本地识别题目文字
- 调用 DeepSeek API 生成答案
- 半透明置顶浮窗显示答案

## 环境要求

- Windows 10/11
- Python 3.10 及以上
- DeepSeek API Key

## 安装步骤

1. 克隆仓库后进入目录
2. 创建虚拟环境：python -m venv venv
3. 激活虚拟环境：venv\Scripts\activate
4. 安装依赖：pip install -r requirements.txt
5. 复制 .env.example 为 .env，填入你的 DeepSeek API Key

## 使用

运行：python app.py

或双击 start.bat

使用流程：
1. 把题目页面放到屏幕左半边
2. 点浮窗上的「开始检测」
3. 答案自动显示
4. 不想刷新时点「停止检测」

## 配置

在 app.py 顶部可调整：
- REGION：监测区域，默认左半边
- INTERVAL：检测间隔，默认 1.5 秒
- CHANGE_THRESHOLD：变化灵敏度，越小越灵敏

## 免责声明

本项目仅供学习交流，请勿用于考试作弊等违规场景。
