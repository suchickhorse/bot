import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import whisperx
import gc
import torch

# ---------- 配置 ----------
device = "cuda"
audio_file = "./resources/钟表拆装.mp4"
model_size = "medium"
batch_size = 8
compute_type = "float16"
language = "zh"

# ---------- 1. 转录 ----------
print("加载转录模型...")
model = whisperx.load_model(model_size, device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language=language)

del model
gc.collect()
torch.cuda.empty_cache()

# ---------- 2. 对齐 ----------
print("加载对齐模型...")
model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=False
)

del model_a
gc.collect()
torch.cuda.empty_cache()

# ---------- 3. 提取句子级时间戳并写入文件 ----------
segments = result["segments"]  # 每个 segment 包含 start, end, text

# 按开始时间排序（一般已有序）
segments.sort(key=lambda x: x["start"])

output_path = "text.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        f.write(f"{start:.2f}\t{end:.2f}\t{text}\n")

print(f"完成！共 {len(segments)} 句话，已保存到 {output_path}")