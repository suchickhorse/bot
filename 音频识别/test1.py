import os
import torch
import torchaudio
import soundfile as sf
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor
)

# ==================== 1. 配置 ====================
MODEL_DIR = "./my_casia_emotion_model"  # 模型保存路径
SAMPLE_RATE = 16000

# 情感类别（与训练时一致）
EMOTIONS = sorted(["angry", "fear", "happy", "neutral", "sad", "surprise"])
NUM_LABELS = len(EMOTIONS)
id2label = {i: e for i, e in enumerate(EMOTIONS)}
label2id = {e: i for i, e in enumerate(EMOTIONS)}

# 检查模型目录是否存在
if not os.path.exists(MODEL_DIR):
    print(f"错误：模型目录不存在！请先运行train2.py训练模型。")
    print(f"模型目录路径：{MODEL_DIR}")
    exit(1)

# ==================== 2. 加载模型和特征提取器 ====================
print("加载模型和特征提取器...")
try:
    # 加载特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
    # 加载模型
    model = HubertForSequenceClassification.from_pretrained(MODEL_DIR)
    # 将模型移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    print("模型和特征提取器加载成功！")
except Exception as e:
    print(f"加载模型失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ==================== 3. 音频预处理函数 ====================
def preprocess_audio(audio_path):
    """预处理音频文件"""
    # 使用soundfile加载音频
    speech, sr = sf.read(audio_path)
    
    # 重采样到目标采样率
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = torch.tensor(speech).unsqueeze(0)
        waveform = resampler(waveform)
        speech = waveform.squeeze().numpy()
    
    # 使用特征提取器提取特征
    inputs = feature_extractor(
        [speech],  # 包装成列表
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=int(SAMPLE_RATE * 5),  # 最长5秒
    )
    
    # 将输入移至与模型相同的设备
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    
    return inputs

# ==================== 4. 预测函数 ====================
def predict_emotion(audio_path):
    """预测音频的情感"""
    # 预处理音频
    inputs = preprocess_audio(audio_path)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    # 获取预测结果
    predicted_label = id2label[predictions.item()]
    
    # 计算各情感的概率
    probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
    emotion_probabilities = {id2label[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
    
    return predicted_label, emotion_probabilities

# ==================== 5. 主函数 ====================
def main():
    print("\n===== 情感识别测试 =====")
    print(f"可用的情感目录: {EMOTIONS}")
    
    # 让用户选择情感目录
    while True:
        emotion_dir = input("请输入情感目录（如：angry, fear, happy等）: ").strip().lower()
        if emotion_dir in EMOTIONS:
            break
        print(f"错误：情感目录不存在，请从以下选项中选择: {EMOTIONS}")
    
    # 构建音频文件路径
    audio_dir = os.path.join("./dataset/CASIA database/liuchanhg", emotion_dir)
    if not os.path.exists(audio_dir):
        print(f"错误：目录不存在: {audio_dir}")
        exit(1)
    
    # 列出该目录下的音频文件
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
    if not audio_files:
        print(f"错误：目录中没有音频文件: {audio_dir}")
        exit(1)
    
    print(f"\n目录 {emotion_dir} 中的音频文件:")
    for i, file in enumerate(audio_files, 1):
        print(f"{i}. {file}")
    
    # 让用户选择音频文件
    while True:
        try:
            file_index = int(input("请输入要测试的文件序号: ")) - 1
            if 0 <= file_index < len(audio_files):
                break
            print(f"错误：请输入有效的序号（1-{len(audio_files)}）")
        except ValueError:
            print("错误：请输入数字序号")
    
    # 获取选定的音频文件路径
    selected_file = audio_files[file_index]
    audio_path = os.path.join(audio_dir, selected_file)
    print(f"\n测试文件: {audio_path}")
    print(f"真实情感: {emotion_dir}")
    
    # 预测情感
    print("\n正在分析...")
    predicted_label, probabilities = predict_emotion(audio_path)
    
    # 显示结果
    print("\n===== 预测结果 =====")
    print(f"预测情感: {predicted_label}")
    print(f"预测{'正确' if predicted_label == emotion_dir else '错误'}")
    
    print("\n各情感概率:")
    for emotion, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {prob}%")

if __name__ == "__main__":
    main()
