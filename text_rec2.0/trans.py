"""
翻译模块 —— 基于 Ollama 本地大模型的文本翻译
==============================================

本模块封装了通过 Ollama 调用本地大模型进行翻译的逻辑。

依赖：
  - ollama（Python 客户端库）
  - 本地已安装并启动的 Ollama 服务
  - 已拉取的目标模型（默认 qwen2.5:7b-instruct-q4_K_M）

架构说明：
  Translator 类采用"系统提示词（System Prompt）+ 用户输入"的模式，
  通过精心设计的模板告诉模型"你是一个翻译助手"，并要求只输出译文。
  这种设计保证了翻译质量的稳定性和输出格式的一致性。

为什么使用 Ollama 本地模型？
  1. 无需联网，保护隐私（数据不离开本机）
  2. 无 API 调用费用
  3. 翻译质量优于传统规则引擎
  4. 内置 OCR 纠错能力（通过系统提示词引导模型自动修正识别错误）

API 文档参考：
  ollama.chat(model, messages, stream)  —— 非流式对话接口
"""

import ollama

# ---------- 默认模型配置 ----------
# qwen2.5:7b-instruct-q4_K_M 是通义千问 2.5 的 70 亿参数版本，
# 使用 4-bit 量化（q4_K_M），在翻译质量与运行效率之间取得平衡。
# 如需更换模型，修改此常量即可，例如：
#   DEFAULT_MODEL = 'llama3.2:3b'          （更轻量）
#   DEFAULT_MODEL = 'qwen2.5:14b'          （更强，但需要更大显存）
DEFAULT_MODEL = 'qwen2.5:7b-instruct-q4_K_M'

# ---------- 系统提示词模板 ----------
# %s 占位符将在运行时替换为具体的翻译指令（如"英→中"或"中→英"）。
#
# 设计考量：
#   "你是一个专业的文本翻译助手" —— 设定角色，引导模型以翻译专家身份工作。
#   "只返回翻译结果，不要添加任何解释、注释或额外信息"
#      —— 约束输出格式，确保结果纯粹是翻译文本，可以直接展示给用户。
#   "如果原文存在可能的OCR识别错误，请在保持原意的前提下自动修正"
#      —— 给模型适当的自由度去修正 OCR 常见错误（如形近字、乱码等）。
#         这利用了 LLM 的语境理解能力来弥补 OCR 引擎的不足。
TRANS = f"""你是一个专业的文本翻译助手。%s，
    只返回翻译结果，不要添加任何解释、注释或额外信息。如果原文存在可能的OCR识别错误，
    请在保持原意的前提下自动修正。
    """


class Translator:
    """
    翻译器类，封装 Ollama 模型调用逻辑。

    属性：
      model         : str —— Ollama 模型名称
      system_prompt : str —— 系统提示词模板（包含 %s 占位符）

    公开方法：
      translate(text, direction) → str  —— 执行翻译并返回译文

    设计模式：
      - 单次翻译即可复用的实例（如需并发翻译，可创建多个实例）
      - 系统提示词在 __init__ 时配置，translate 时注入具体翻译方向
    """

    def __init__(self, model=DEFAULT_MODEL, system_prompt=TRANS):
        """
        初始化翻译器。

        参数：
          model         : str —— Ollama 模型名称（默认见 DEFAULT_MODEL）
          system_prompt : str —— 系统提示词模板（默认见 TRANS），
                                其中 %s 将被替换为翻译方向描述
        """
        self.model = model
        self.system_prompt = system_prompt

    def translate(self, text, direction='en-zh') -> str:
        """
        执行翻译。

        参数：
          text      : str —— 待翻译的文本。如果为空字符串，直接返回空字符串。
          direction : str —— 翻译方向：
                            'en-zh'  —— 英文翻译为中文（默认值）
                            'zh-en'  —— 中文翻译为英文

        返回：
          str —— 翻译结果文本，或空字符串（如果输入为空）

        执行流程：
          1. 空文本检查 → 直接返回 ''（避免浪费一次模型调用）
          2. 根据 direction 确定翻译模式描述
          3. 将模式描述注入系统提示词模板（%s 占位符替换）
          4. 构造 messages 列表：
             [{'role': 'system', 'content': 系统提示词},
              {'role': 'user',   'content': 待翻译文本}]
          5. 调用 ollama.chat()（非流式，stream=False）
          6. 从返回结构中提取 response['message']['content']

        ollama.chat 返回值结构：
          {
            'model': 'qwen2.5:7b-instruct-q4_K_M',
            'message': {
              'role': 'assistant',
              'content': '译文内容...'
            },
            'done': True,
            ...
          }

        错误处理：
          不在本层做 try/except，将异常向上抛出，
          由调用方（main.py 中的 TranslationWorker.run()）捕获并通过信号传递到 UI。
        """
        if text:
            # 根据翻译方向生成具体的指令字符串
            if direction == 'zh-en':
                mode = '请把文字中的中文翻译成英文'
            else:
                mode = '请把文字中的英文翻译成中文'

            # 调用 Ollama 对话接口
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt % mode},
                    {'role': 'user',   'content': text}
                ],
                stream=False  # 非流式：等待完整结果后返回
            )

            # 提取并返回助手回复的文本内容
            return response['message']['content']
        else:
            return ''


# ============================================================================
# 模块自行测试（直接执行 trans.py 时运行）
# ============================================================================
if __name__ == '__main__':
    trans_instance = Translator()
    response = trans_instance.translate('人民万岁')  # 中文 → 英文
    print(response)
