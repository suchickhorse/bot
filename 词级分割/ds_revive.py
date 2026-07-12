import os
import whisperx
import gc
import torch

# ========== 强制离线，只从本地加载 ==========
os.environ["HF_HUB_OFFLINE"] = "1"
# 设置 Hugging Face 缓存根目录（你的 models/huggingface 文件夹）
os.environ["HF_HOME"] = "G:/PythonProject/Whisper/models/huggingface"

if __name__ == '__main__':
    # ----- 配置 -----
    device = "cuda"
    compute_type = "float16"          # GPU 推荐用 float16，节省显存
    language = "zh"
    audio_file = "./resources/test_dialogue.mp3"   # 改成你真实的音频文件路径
    prompt = "下班后的日常对话"
    download_root = "G:/PythonProject/Whisper/models/huggingface/hub"  # 直接指向 hub 文件夹

    # 1. 加载转录模型（完全本地）
    model = whisperx.load_model(
        "large-v2",
        device=device,
        compute_type=compute_type,
        download_root=download_root,
        language=language
    )

    # 2. 加载音频
    audio = whisperx.load_audio(audio_file)

    # 3. 转录（注意：不要传入 model）
    result = model.transcribe(
        audio,
        batch_size=8,
        language=language
    )

    # 释放转录模型（可选，节省显存）
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 4. 加载对齐模型（也会从本地 HF_HOME 下加载）
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)

    # 5. 对齐
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # 释放对齐模型
    del align_model
    gc.collect()
    torch.cuda.empty_cache()

    # 6. 输出结果
    for seg in result["segments"]:
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")