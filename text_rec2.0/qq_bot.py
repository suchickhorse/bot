"""
QQ 自动回复机器人 —— 主程序
==========================

功能：
  每隔约 5 秒截取 QQ 聊天窗口 → OCR 识别 → 检测新消息 → AI 生成回复 → 模拟键盘发送。

工作流程：
  1. 找到 QQ 聊天窗口
  2. 截取聊天消息区域
  3. OCR 识别文字
  4. 与上次结果比较，判断是否有新消息
  5. 如有对方新消息 → 调用 AI 生成回复
  6. 在 QQ 输入框中输入回复并按回车发送
  7. 等待冷却，进入下一轮

使用方法：
  conda activate text_rec
  python qq_bot.py

退出：
  按 Ctrl+C 优雅退出。

依赖：
  - PaddleOCR（PP-OCRv5 模型，路径 ../models/）
  - Ollama 服务已启动 + qwen2.5:7b-instruct-q4_K_M
  - keyboard 包（已安装在 text_rec 环境）
  - 本项目的 config.py / qq_capture.py / reply_engine.py
"""

import sys
import os
import hashlib
import time
import signal
from collections import deque
from difflib import SequenceMatcher

import numpy as np

# ---------- keyboard（用于模拟键盘发送消息） ----------
# 注意：Windows 上需要管理员权限才能模拟键盘事件，
# 如果 QQ 以管理员身份运行，本脚本也需要管理员权限。
try:
    import keyboard
except ImportError:
    print("错误：缺少 keyboard 包，请运行: pip install keyboard")
    sys.exit(1)

# ---------- PaddleOCR ----------
from paddleocr import PaddleOCR

# ---------- 本项目的模块 ----------
import config
from qq_capture import FixedRegionCapture
from reply_engine import ReplyEngine


# ============================================================================
# 一、OCR 引擎初始化（全局单例，启动时加载一次）
# ============================================================================
print("正在加载 OCR 模型，请稍候...")
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='ch',
    text_detection_model_dir='../models/PP-OCRv5_server_det_infer',
    text_recognition_model_dir='../models/PP-OCRv5_server_rec_infer',
    textline_orientation_model_dir='../models/PP-LCNet_x1_0_textline_ori_infer',
)
print("OCR 模型就绪！")


# ============================================================================
# 二、QQ 自动回复机器人
# ============================================================================
class QQAutoReplyBot:
    """
    主机器人控制器。

    状态变量（跨周期持久化）：
        last_image_md5   : str          —— 上次截图的 MD5
        last_ocr_text    : str          —— 上次 OCR 识别的完整文字
        sent_hashes      : set[str]     —— 已发送回复的 MD5 集合（精确去重）
        recent_sent_texts: deque[str]   —— 最近发送的回复文本（模糊去重）
        conversation_buf : deque[(str,str)] —— 对话上下文缓冲区
        running          : bool         —— 运行标志（Ctrl+C 设为 False）
    """

    def __init__(self):
        """初始化各模块和状态变量。"""
        # ---- 窗口截取模块（固定坐标） ----
        self.capture = FixedRegionCapture()

        # ---- AI 回复模块 ----
        self.reply_engine = ReplyEngine(
            backend=config.AI_BACKEND,
            api_key=config.DEEPSEEK_API_KEY,
            api_url=config.DEEPSEEK_API_URL,
            model=config.DEEPSEEK_MODEL,
            system_prompt=config.PERSONALITY_SYSTEM_PROMPT,
            max_length=config.MAX_REPLY_LENGTH_CHARS,
            timeout=config.AI_TIMEOUT_SECONDS,
            fallback_reply=config.FALLBACK_REPLY,
        )

        # ---- 状态变量 ----
        self.last_image_md5 = None         # 上次截图的 MD5
        self.last_ocr_text = ""            # 上次 OCR 完整文本
        self.sent_hashes = set()           # 已发送回复的 MD5 集合
        self.recent_sent_texts = deque(maxlen=config.RECENT_SENT_TEXTS_SIZE)
        self.conversation_buf = deque(maxlen=config.CONVERSATION_BUFFER_SIZE * 2)
        self.running = True

    # ======================================================================
    # 主循环
    # ======================================================================
    def run(self):
        """
        机器人主循环。

        每个周期：
          截图 → MD5快速去重 → OCR → 文本去重 → 提取新内容 →
          过滤自回复 → AI生成 → 发送 → 冷却 → 下一周期
        """
        self._log("QQ 自动回复机器人已启动")
        self._log(f"检查间隔: {config.CHECK_INTERVAL_SECONDS} 秒")
        self._log(f"AI 后端: {config.AI_BACKEND} ({config.DEEPSEEK_MODEL if config.AI_BACKEND == 'deepseek' else config.OLLAMA_MODEL})")
        self._log("按 Ctrl+C 退出\n")

        while self.running:
            try:
                self._one_cycle()
            except Exception as e:
                self._log(f"周期异常: {e}")
                self._sleep(config.CHECK_INTERVAL_SECONDS)

        self._log("机器人已退出。")

    def _one_cycle(self):
        """执行一个完整的检查周期。"""
        # ---- 第 1 步：截取固定区域 ----
        img = self.capture.capture()
        if img is None:
            self._sleep(config.CHECK_INTERVAL_SECONDS)
            return

        # ---- 第 3 步：图像 MD5 快速去重 ----
        img_md5 = hashlib.md5(img.tobytes()).hexdigest()
        if img_md5 == self.last_image_md5:
            # 像素完全相同，绝对无变化
            self._sleep(config.CHECK_INTERVAL_SECONDS)
            return

        # ---- 第 4 步：OCR 识别 ----
        current_ocr_text = self._ocr_image(img)
        if current_ocr_text is None:
            self._sleep(config.CHECK_INTERVAL_SECONDS)
            return

        # ---- 第 5 步：文本去重 ----
        if current_ocr_text.strip() == self.last_ocr_text.strip():
            # 文本无变化（可能只是细微渲染差异）
            self.last_image_md5 = img_md5
            self._sleep(config.CHECK_INTERVAL_SECONDS)
            return

        # ---- 第 6 步：提取新内容 ----
        new_text = self._extract_new_content(self.last_ocr_text, current_ocr_text)

        # 更新状态
        self.last_image_md5 = img_md5
        self.last_ocr_text = current_ocr_text

        if not new_text or not new_text.strip():
            self._sleep(config.CHECK_INTERVAL_SECONDS)
            return

        self._log(f"检测到新文字:\n  {new_text[:100]}...")

        # ---- 第 7 步：过滤自己的回复 ----
        if self._is_own_reply(new_text):
            self._log("  (识别为自己的回复，跳过)")
            self._sleep(config.CHECK_INTERVAL_SECONDS)
            return

        # ---- 第 8 步：确认为对方新消息 → AI 生成回复 ----
        self._log("  → 对方新消息，生成回复中...")
        self.conversation_buf.append(("对方", new_text.strip()))

        reply = self.reply_engine.generate_reply(
            new_text,
            conversation_history=list(self.conversation_buf)
        )

        if not reply:
            self._log("  AI 未生成回复，跳过")
            self._sleep(config.CHECK_INTERVAL_SECONDS)
            return

        self._log(f"  AI 回复: {reply[:80]}...")

        # ---- 第 9 步：发送回复 ----
        success = self._send_reply(reply)

        # ---- 第 10 步：更新发送状态 ----
        if success:
            reply_stripped = reply.strip()
            reply_hash = hashlib.md5(reply_stripped.encode()).hexdigest()
            self.sent_hashes.add(reply_hash)
            self.recent_sent_texts.append(reply_stripped)
            self.conversation_buf.append(("你", reply_stripped))

            self._log("  发送成功！")
            # 冷却：等待 QQ 渲染已发送的消息
            self._sleep(config.POST_SEND_COOLDOWN_SECONDS)
        else:
            self._log("  发送失败！")

        # ---- 等待下一周期 ----
        self._sleep(config.CHECK_INTERVAL_SECONDS)

    # ======================================================================
    # 子步骤方法
    # ======================================================================

    def _ocr_image(self, img):
        """
        对 PIL Image 执行 OCR 识别，返回拼接后的文字。

        返回：
            str  —— 识别出的所有文字，每行一条
            None —— 识别失败
        """
        try:
            img_arr = np.array(img)
            results = ocr.predict(img_arr)

            lines = []
            for res in results:
                for t in res.get('rec_texts', []):
                    lines.append(t)

            return "\n".join(lines) if lines else ""
        except Exception as e:
            self._log(f"OCR 识别出错: {e}")
            return None

    def _extract_new_content(self, old_text, new_text):
        """
        从新旧两次 OCR 结果中提取新增的文字。

        算法：基于行的最长公共后缀（Longest Common Suffix）。
        因为 QQ 消息按时间从旧到新排列在聊天区域中（新消息在底部），
        新增的文字总是出现在 OCR 文本的末尾。

        步骤：
          1. 按行分割新旧文本
          2. 从末尾向前找最长公共后缀
          3. 新文本去掉公共后缀即为新增内容
          4. 如果新增行数过多（>20行），视为页面滚动而非新消息

        返回：
            str —— 新增文字，如果没有则返回 ""
        """
        old_lines = [l.strip() for l in old_text.strip().split('\n') if l.strip()]
        new_lines = [l.strip() for l in new_text.strip().split('\n') if l.strip()]

        if not new_lines:
            return ""

        # 找最长公共后缀：从末尾向前比较
        common_suffix_len = 0
        max_common = min(len(old_lines), len(new_lines))
        for i in range(1, max_common + 1):
            if old_lines[-i] == new_lines[-i]:
                common_suffix_len += 1
            else:
                break

        # 新行 = 去掉公共后缀的前缀部分
        new_line_count = len(new_lines) - common_suffix_len

        if new_line_count <= 0:
            return ""

        # 滚动检测：新增行数过多 → 可能是页面滚动
        if new_line_count > config.MAX_NEW_LINES_THRESHOLD:
            return ""

        new_only_lines = new_lines[:new_line_count]
        return '\n'.join(new_only_lines)

    def _is_own_reply(self, text):
        """
        判断 OCR 识别出的文字是否是机器人自己刚发送的回复（被截图误截）。

        多层检测（任一命中即判定为自己的消息）：
          1. 精确 MD5 匹配
          2. 整体文本模糊匹配（SequenceMatcher）
          3. 子串包含检测（自己的回复是否出现在 OCR 文本中）
          4. 字符级重叠率检测（处理 OCR 乱码拆字的情况）
        """
        text_stripped = text.strip()
        if not text_stripped:
            return True

        # 1. 精确 MD5 匹配
        text_hash = hashlib.md5(text_stripped.encode()).hexdigest()
        if text_hash in self.sent_hashes:
            return True

        # 遍历最近发送的每条回复，逐条比对
        for sent_text in self.recent_sent_texts:
            sent_stripped = sent_text.strip()
            if not sent_stripped:
                continue

            # 2. 整体模糊匹配
            similarity = SequenceMatcher(None, text_stripped, sent_stripped).ratio()
            if similarity > config.SELF_REPLY_SIMILARITY_THRESHOLD:
                return True

            # 3. 子串包含检测
            #    自己的回复较长（>5字）且 OCR 文本中包含它 → 自己的消息
            if len(sent_stripped) >= 5 and sent_stripped in text_stripped:
                return True
            #    OCR 文本较长且被自己的回复包含 → 自己的消息被截断识别
            if len(text_stripped) >= 5 and text_stripped in sent_stripped:
                return True

            # 4. 字符级重叠率（处理 OCR 拆字/乱码）
            #    将两个文本拆成字符集，计算交集占比
            text_chars = set(text_stripped.replace(' ', ''))
            sent_chars = set(sent_stripped.replace(' ', ''))
            if text_chars and sent_chars:
                char_overlap = len(text_chars & sent_chars) / len(text_chars)
                # 如果 OCR 文本 80% 以上的字符都出现在自己回复中 →
                # 极大概率是自己的消息被 OCR 误读
                if char_overlap > 0.80:
                    return True

        return False

    def _send_reply(self, reply_text):
        """
        在 QQ 窗口中输入并发送回复（QQ 窗口已打开，鼠标已在输入框）。

        步骤：
          1. 原位点击鼠标让 QQ 获得焦点
          2. Ctrl+A + Backspace 清空已有文字
          3. 逐字符输入回复内容
          4. 按 Enter 发送

        返回：
            bool —— 是否成功
        """
        try:
            self._log(f"  发送: {reply_text[:50]}...")

            # 1. 在鼠标当前位置点击左键，让 QQ 输入框获得键盘焦点
            #    不移动光标，只在原位点击（用户鼠标本来就在输入框上）
            self.capture.click_at_current_position()

            # 2. 清空输入框已有内容
            keyboard.press_and_release('ctrl+a')
            time.sleep(0.05)
            keyboard.press_and_release('backspace')
            time.sleep(0.05)

            # 3. 输入回复
            keyboard.write(reply_text, delay=0.02)
            time.sleep(0.1)

            # 4. 按发送快捷键
            send_key = config.SEND_KEY
            if send_key == 'ctrl+enter':
                keyboard.press('ctrl')
                keyboard.press_and_release('enter')
                keyboard.release('ctrl')
            else:
                keyboard.press_and_release('enter')
            time.sleep(0.1)

            self._log("  发送完成")
            return True
        except Exception as e:
            self._log(f"  发送出错: {e}")
            return False

    # ======================================================================
    # 工具方法
    # ======================================================================

    def _log(self, msg):
        """打印日志（可通过 config.ENABLE_CONSOLE_LOG 控制）。"""
        if config.ENABLE_CONSOLE_LOG:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")

    def _sleep(self, seconds):
        """
        可中断的 sleep。
        如果 running 变为 False（Ctrl+C），立即返回。
        将长 sleep 拆分为 0.5 秒小段，确保响应及时。
        """
        elapsed = 0.0
        while elapsed < seconds and self.running:
            chunk = min(0.5, seconds - elapsed)
            time.sleep(chunk)
            elapsed += chunk

    def stop(self):
        """停止机器人运行。"""
        self.running = False


# ============================================================================
# 三、程序入口
# ============================================================================
if __name__ == "__main__":
    bot = QQAutoReplyBot()

    # 注册 SIGINT 处理器（Ctrl+C）
    def sigint_handler(sig, frame):
        print("\n收到退出信号，正在关闭...")
        bot.stop()

    signal.signal(signal.SIGINT, sigint_handler)

    # 启动机器人
    bot.run()
