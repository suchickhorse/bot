from snownlp import SnowNLP

text = "今天阳光真好啊！"
s = SnowNLP(text)
score = s.sentiments  # 0.99

if score > 0.6:
    label = "积极"
elif score < 0.4:
    label = "消极"
else:
    label = "中立"