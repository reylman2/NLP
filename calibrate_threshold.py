# calibrate_threshold.py

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification
from train import CustomToxicClassifier, load_data, prepare_training_data

def main():
    # 1) Load & split data exactly as in train.py
    df = load_data()
    texts, labels = prepare_training_data(df)
    _, val_texts, _, val_labels = train_test_split(
        texts, labels,
        test_size=0.1,
        random_state=42,
        stratify=labels
    )

    # 2) Load tokenizer & sequence-classification model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    seq_model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    # Wrap in your custom head
    model = CustomToxicClassifier(seq_model)
    ckpt = torch.load('model/model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3) Compute toxic probabilities on val set
    probs = []
    for txt in val_texts:
        enc = tokenizer(
            txt,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        with torch.no_grad():
            logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            prob = torch.softmax(logits, dim=1)[0,1].item()
        probs.append(prob)

    # 4) Find best threshold by max F1
    precision, recall, thresholds = precision_recall_curve(val_labels, probs)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    print(f'Best threshold = {best_thresh:.3f}, F1 = {best_f1:.3f}')

if __name__ == '__main__':
    main()
