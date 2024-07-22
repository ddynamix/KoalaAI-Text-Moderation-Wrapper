from typing import List, Tuple

from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = snapshot_download(repo_id="KoalaAI/Text-Moderation")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def predict(text: str) -> List[Tuple[str, float]]:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).squeeze()
    id2label = model.config.id2label
    labels = [id2label[idx] for idx in range(len(probabilities))]
    label_prob_pairs = list(zip(labels, probabilities.tolist()))
    label_prob_pairs.sort(key=lambda item: item[1], reverse=True)
    return label_prob_pairs
