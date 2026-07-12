import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

texts = [
    "今天阳光真好啊！",
    "唉，又被领导批评了，难受",
    "中午吃啥？随便吧",
    "今天天气不错，适合出去玩"
]

for text in texts:
    result = classifier(text)[0]  # 返回字典
    print(f"文本: {text}")
    print(f"情绪: {result['label']} (置信度: {result['score']:.2f})\n")