"""
AI 回复生成模块
==============

支持两种后端：
  - deepseek : DeepSeek API（云端 V3，速度快）
  - ollama   : 本地 Ollama 模型（已注释，如需使用设置 AI_BACKEND='ollama'）

API 文档：https://api-docs.deepseek.com/
"""

import json
import threading
import urllib.request
import urllib.error


# ============================================================================
# DeepSeek API 调用
# ============================================================================

def _call_deepseek_api(messages, api_key, api_url, model, timeout):
    """
    调用 DeepSeek API（OpenAI 兼容格式），在独立线程中执行。

    返回：
        str  —— AI 回复内容
        None —— 超时或出错
    """
    result_holder = [None]
    error_holder = [None]

    def _call():
        try:
            body = json.dumps({
                'model': model,
                'messages': messages,
                'stream': False,
                'max_tokens': 512,
            }).encode('utf-8')

            req = urllib.request.Request(api_url, data=body)
            req.add_header('Authorization', f'Bearer {api_key}')
            req.add_header('Content-Type', 'application/json')

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                result_holder[0] = data['choices'][0]['message']['content']

        except Exception as e:
            error_holder[0] = str(e)

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"[DeepSeek] API 调用超时 ({timeout}s)")
        return None

    if error_holder[0]:
        print(f"[DeepSeek] API 调用出错: {error_holder[0]}")
        return None

    return result_holder[0] or ""


# ============================================================================
# ReplyEngine 类
# ============================================================================

class ReplyEngine:
    """
    AI 聊天回复生成器。

    支持 DeepSeek API 和本地 Ollama 两种后端。

    用法示例：
        engine = ReplyEngine(
            backend='deepseek',
            api_key='sk-xxx',
            model='deepseek-chat',
        )
        reply = engine.generate_reply("今天天气真好", conversation_history=[
            ("对方", "你好呀"),
            ("你", "你好！"),
        ])
    """

    def __init__(self, backend='deepseek',
                 api_key=None, api_url=None, model=None,
                 system_prompt=None, max_length=200, timeout=30,
                 fallback_reply="嗯嗯"):
        """
        参数：
            backend        : str  —— 'deepseek' 或 'ollama'
            api_key        : str  —— DeepSeek API Key
            api_url        : str  —— API 地址
            model          : str  —— 模型名称
            system_prompt  : str  —— 系统提示词模板
            max_length     : int  —— 回复最大字符数
            timeout        : int  —— API 超时秒数
            fallback_reply : str  —— 超时/出错时的默认回复
        """
        self.backend = backend
        self.api_key = api_key or ''
        self.api_url = api_url or 'https://api.deepseek.com/chat/completions'
        self.model = model or 'deepseek-chat'
        self.system_prompt = system_prompt or (
            '你是一个正在QQ上和朋友聊天的普通用户。'
            '你的回复应该自然、口语化，像真人聊天一样。'
            '回复要简洁（一般不超过100字），可以适当使用表情符号和网络用语。'
            '根据对方的消息内容进行有意义的回复，不要机械重复对方的话。'
            '如果你不确定如何回复，可以表达兴趣或问一个相关问题。'
            '绝对不要在回复中暴露你是AI或机器人。'
            '只返回回复内容本身，不要添加任何前缀、后缀或解释。'
            '\n\n最近的对话记录（供上下文参考）：\n{conversation_context}'
        )
        self.max_length = max_length
        self.timeout = timeout
        self.fallback_reply = fallback_reply

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def generate_reply(self, new_message, conversation_history=None):
        """根据新消息和对话历史，生成 AI 回复。"""
        if not new_message or not new_message.strip():
            return ""

        # 1. 构建上下文文本
        context_text = self._build_context(conversation_history)

        # 2. 注入系统提示词
        system_content = self.system_prompt.replace(
            "{conversation_context}", context_text
        )

        # 3. 构建消息列表
        user_prompt = (
            f'对方刚刚发来消息：「{new_message}」\n\n'
            f'请你根据上面的对话记录和对方的最新消息，'
            f'以你的性格设定，给出一条自然、有趣的回复。'
        )
        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user',   'content': user_prompt},
        ]

        # 4. 调用 AI 后端
        if self.backend == 'deepseek':
            result = _call_deepseek_api(
                messages, self.api_key, self.api_url,
                self.model, self.timeout
            )
        else:
            # Ollama（保留兼容，需 import ollama）
            result = self._call_ollama(messages)

        # 5. 后处理
        if result is None:
            return self.fallback_reply

        reply = result.strip()
        if len(reply) > self.max_length:
            reply = reply[:self.max_length]

        return reply

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_context(self, conversation_history):
        """将对话历史格式化为文本上下文。"""
        if not conversation_history:
            return "（暂无对话记录）"

        lines = []
        for role, text in conversation_history:
            text = text.replace('\n', ' ').strip()
            if text:
                lines.append(f"{role}: {text}")

        return "\n".join(lines) if lines else "（暂无对话记录）"

    # ------------------------------------------------------------------
    # Ollama 后端（保留兼容，需 import ollama）
    # ------------------------------------------------------------------
    def _call_ollama(self, messages):
        """本地 Ollama 调用（废弃，保留备用）。"""
        try:
            import ollama
        except ImportError:
            print("[ReplyEngine] ollama 未安装，无法使用本地模型")
            return None

        result_holder = [None]
        error_holder = [None]

        def _call():
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                )
                result_holder[0] = response['message']['content']
            except Exception as e:
                error_holder[0] = str(e)

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            print(f"[Ollama] 调用超时 ({self.timeout}s)")
            return None

        if error_holder[0]:
            print(f"[Ollama] 调用出错: {error_holder[0]}")
            return None

        return result_holder[0] or ""


# ============================================================================
# 模块自行测试
# ============================================================================
if __name__ == "__main__":
    print("=== AI 回复引擎测试 ===\n")

    # 测试需要有效的 API Key
    engine = ReplyEngine(
        backend='deepseek',
        api_key='your-api-key-here',
    )

    test_messages = ["你好呀", "今天天气真好", "周末有什么安排？"]

    for msg in test_messages:
        print(f">>> 对方: {msg}")
        reply = engine.generate_reply(msg)
        print(f"    AI:  {reply}\n")
