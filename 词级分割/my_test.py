import os
os.environ["HF_HOME"] = "./models/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
import whisperx
import torch

if __name__ == '__main__':
    model_name = "large-v2"
    device = "cuda"
    compute_type = "float32"
    language = "zh"
    audio_path = "./resources/test_dialogue.mp3"
    prompt = "下班后的日常对话"
    download_root = "./models/huggingface/hub"

    # 加载转录模型
    model = whisperx.load_model("large-v2",
                       device = device,
                       compute_type=compute_type,
                       download_root=download_root,
                       language="zh")

    # 加载音频文件
    audio = whisperx.load_audio(audio_path)

    # 转录音频
    result = model.transcribe(audio)

    # 加载对齐模型
    align_model,metadata = whisperx.load_align_model(language_code="zh", device=device)

    # 进行对齐
    result = whisperx.align(result["segments"],align_model,metadata,audio,device)

    for segment in result["segments"]:
        print(segment)