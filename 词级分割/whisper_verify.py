import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import whisperx
import gc
import torch
from whisperx.diarize import DiarizationPipeline

device = "cuda"
audio_file = "./resources/钟表拆装.mp4"
batch_size = 8          # 调低一些以防显存不足
compute_type = "float16"

# 1. 转录（指定语言为中文）
model = whisperx.load_model("medium", device, compute_type=compute_type)  # 或 "large-v2"
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="zh")

# 释放转录模型
del model
gc.collect()
torch.cuda.empty_cache()

# 2. 对齐（强制用中文的对齐模型）
model_a, metadata = whisperx.load_align_model(language_code="zh", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

del model_a
gc.collect()
torch.cuda.empty_cache()

# 3. 说话人分离（假设说话人数 1~2 人）
token = os.getenv("HF_WHISPERX")
if token is None:
    raise ValueError("请先设置环境变量 HF_WHISPERX")
diarize_model = DiarizationPipeline(token=token, device=device)
diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=2)
result = whisperx.assign_word_speakers(diarize_segments, result)

print(result["segments"])