import os
import torch
import torchaudio
import soundfile as sf
from datasets import Dataset, DatasetDict
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import numpy as np

# ==================== 1. 配置 ====================
DATA_ROOT = "./dataset/CASIA database"               # CASIA 数据集根目录
OUTPUT_DIR = "./my_casia_emotion_model"       # 模型保存路径（与之前训练的模型路径相同）
SAMPLE_RATE = 16000
BATCH_SIZE = 16
EPOCHS = 1  # 继续训练的轮数
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

# 情感类别（与你的 CASIA 文件夹名严格一致）
EMOTIONS = sorted(["angry", "fear", "happy", "neutral", "sad", "surprise"])
NUM_LABELS = len(EMOTIONS)
id2label = {i: e for i, e in enumerate(EMOTIONS)}
label2id = {e: i for i, e in enumerate(EMOTIONS)}

# 检查模型目录是否存在
if not os.path.exists(OUTPUT_DIR):
    print(f"错误：模型目录不存在！请先运行train2.py训练基础模型。")
    print(f"模型目录路径：{OUTPUT_DIR}")
    exit(1)

# ==================== 2. 从文件夹加载数据 ====================
def load_data(root):
    audio_paths, labels = [], []
    # 遍历所有说话人目录
    for speaker in os.listdir(root):
        speaker_dir = os.path.join(root, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        # 遍历该说话人下的情感文件夹
        for emotion in EMOTIONS:
            emotion_dir = os.path.join(speaker_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for file in os.listdir(emotion_dir):
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_paths.append(os.path.join(emotion_dir, file))
                    labels.append(label2id[emotion])
    return audio_paths, labels

audio_paths, labels = load_data(DATA_ROOT)
print(f"总样本数: {len(audio_paths)}")

# 检查是否加载到样本
if len(audio_paths) == 0:
    print("错误：没有加载到任何样本！")
    print(f"请检查 DATA_ROOT 路径是否正确：{DATA_ROOT}")
    print("确保路径包含正确的说话人目录和情感子目录")
    exit(1)

# 划分 80/20 训练/验证集（保持类别比例）
train_paths, val_paths, train_labels, val_labels = train_test_split(
    audio_paths, labels, test_size=0.2, random_state=SEED, stratify=labels
)

train_dataset = Dataset.from_dict({"audio": train_paths, "label": train_labels})
val_dataset = Dataset.from_dict({"audio": val_paths, "label": val_labels})
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

# ==================== 3. 特征提取 ====================
# 从保存的模型目录加载特征提取器
print("从保存的模型目录加载特征提取器...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(OUTPUT_DIR)
print("特征提取器加载成功！")

def preprocess(batch):
    speech_list = []
    for path in batch["audio"]:
        # 使用soundfile加载音频文件
        speech, sr = sf.read(path)
        if sr != SAMPLE_RATE:
            # 使用torchaudio进行重采样
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = torch.tensor(speech).unsqueeze(0)
            waveform = resampler(waveform)
            speech = waveform.squeeze().numpy()
        speech_list.append(speech)

    inputs = feature_extractor(
        speech_list,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=int(SAMPLE_RATE * 5),  # 最长5秒，可调整
    )
    inputs["labels"] = torch.tensor(batch["label"], dtype=torch.long)
    return inputs

encoded_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ==================== 4. 模型 ====================
# 从保存的模型目录加载模型
print("从保存的模型目录加载模型...")
model = HubertForSequenceClassification.from_pretrained(
    OUTPUT_DIR,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
print("模型加载成功！")

# ==================== 5. 训练配置 ====================
print("正在创建TrainingArguments...")
try:
    # 尝试使用不同的评估策略参数
    try:
        # 尝试使用evaluation_strategy
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=2,
            seed=SEED,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        print("使用evaluation_strategy成功！")
    except TypeError:
        # 尝试使用eval_strategy
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=2,
            seed=SEED,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        print("使用eval_strategy成功！")
    print("TrainingArguments创建成功！")
except Exception as e:
    print(f"创建TrainingArguments失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ==================== 6. 开始训练 ====================
print("开始继续训练模型...")
trainer.train()

# 保存最终模型及特征提取器
print("保存模型...")
trainer.save_model(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
print(f"训练完成，模型已保存至：{OUTPUT_DIR}")
