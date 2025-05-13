import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import kaggle
import json
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import random

class CustomToxicClassifier(nn.Module):
    def __init__(self, base_model, num_labels=2, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last two layers
        for layer in self.base_model.distilbert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
            
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token output
        logits = self.classifier(pooled_output)
        return logits

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64, augment=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def augment_text(self, text):
        """Augment text data with various techniques"""
        if not self.augment or random.random() > 0.3:  # 30% chance to augment
            return text
            
        # Randomly select one augmentation method
        method = random.choice(['leet', 'spaces', 'punctuation', 'repeat'])
        
        if method == 'leet':
            # Leetspeak conversion
            leet_map = {
                'a': '4', 'e': '3', 'i': '1', 'o': '0',
                's': '5', 't': '7', 'b': '8'
            }
            for char, replacement in leet_map.items():
                if random.random() < 0.3:  # 30% chance to replace character
                    text = text.replace(char, replacement)
                    
        elif method == 'spaces':
            # Add or remove spaces
            if random.random() < 0.5:
                text = text.replace(' ', '')
            else:
                text = ' '.join(text)
                
        elif method == 'punctuation':
            # Add punctuation
            puncts = ['!', '?', '.', ',', ';', ':']
            if random.random() < 0.3:
                text = text + random.choice(puncts)
                
        elif method == 'repeat':
            # Repeat characters
            if len(text) > 2:
                pos = random.randint(0, len(text)-1)
                text = text[:pos] + text[pos] + text[pos:]
        
        return text

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Apply data augmentation
        if self.augment:
            text = self.augment_text(text)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def download_twitter_dataset():
    """Download Twitter Hate and Offensive Language dataset"""
    if not os.path.exists('data/twitter_hate.csv'):
        print("Downloading Twitter dataset...")
        url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
        df = pd.read_csv(url)
        # Combine hate_speech and offensive_language into toxic
        df['toxic'] = ((df['hate_speech'] > 0) | (df['offensive_language'] > 0)).astype(int)
        df = df[['tweet', 'toxic']]
        df.columns = ['comment_text', 'toxic']
        df.to_csv('data/twitter_hate.csv', index=False)
    return pd.read_csv('data/twitter_hate.csv')

def download_waseem_dataset():
    """Download Waseem & Hovy Hate Speech dataset"""
    if not os.path.exists('data/waseem_hate.csv'):
        print("Downloading Waseem dataset...")
        try:
            url = "https://raw.githubusercontent.com/zeerakw/hatespeech/master/NAACL_SRW_2016.csv"
            df = pd.read_csv(url)
            
            # Check and rename columns
            if 'Tweet' in df.columns and 'Label' in df.columns:
                df = df.rename(columns={'Tweet': 'comment_text', 'Label': 'class'})
            elif 'tweet' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'tweet': 'comment_text', 'label': 'class'})
            else:
                print("Warning: Unexpected column names in Waseem dataset. Skipping...")
                return pd.DataFrame(columns=['comment_text', 'toxic'])
            
            # Combine hate_speech and offensive_language into toxic
            df['toxic'] = ((df['class'] == 'hate') | (df['class'] == 'offensive')).astype(int)
            df = df[['comment_text', 'toxic']]
            df.to_csv('data/waseem_hate.csv', index=False)
        except Exception as e:
            print(f"Error downloading Waseem dataset: {str(e)}")
            return pd.DataFrame(columns=['comment_text', 'toxic'])
    return pd.read_csv('data/waseem_hate.csv')

def download_jigsaw_dataset():
    """Download Jigsaw dataset"""
    if not os.path.exists('data/toxic_comments.csv'):
        print("Downloading Jigsaw dataset...")
        kaggle.api.dataset_download_files(
            'julian3833/jigsaw-toxic-comment-classification-challenge',
            path='data',
            unzip=True
        )
        os.rename('data/train.csv', 'data/toxic_comments.csv')
    return pd.read_csv('data/toxic_comments.csv')

def load_data():
    """Load and combine all datasets"""
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load datasets
    jigsaw_df = download_jigsaw_dataset()
    twitter_df = download_twitter_dataset()
    waseem_df = download_waseem_dataset()
    
    # Process Jigsaw data
    jigsaw_df['toxic'] = (jigsaw_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0).astype(int)
    jigsaw_df = jigsaw_df[['comment_text', 'toxic']]
    
    # Combine datasets, handle empty datasets
    dfs_to_concat = []
    
    if not jigsaw_df.empty:
        dfs_to_concat.append(jigsaw_df.sample(n=min(30000, len(jigsaw_df)), random_state=42))
    if not twitter_df.empty:
        dfs_to_concat.append(twitter_df.sample(n=min(10000, len(twitter_df)), random_state=42))
    if not waseem_df.empty:
        dfs_to_concat.append(waseem_df.sample(n=min(10000, len(waseem_df)), random_state=42))
    
    if not dfs_to_concat:
        raise ValueError("No valid datasets available for training")
    
    df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total samples in combined dataset: {len(df)}")
    print(f"Toxic samples: {df['toxic'].sum()}")
    print(f"Non-toxic samples: {len(df) - df['toxic'].sum()}")
    
    return df[['comment_text', 'toxic']]

def plot_confusion_matrix(y_true, y_pred, save_path='visualizations'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_score, save_path='visualizations'):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/roc_curve.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_score, save_path='visualizations'):
    """Plot precision-recall curve"""
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/precision_recall_curve.png')
    plt.close()

def plot_training_metrics(train_losses, val_accuracies, save_path='visualizations'):
    """Plot training metrics (loss and accuracy)"""
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Time')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/training_metrics.png')
    plt.close()

def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def prepare_training_data(df):
    """Prepare training data"""
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = clean_text(str(row['comment_text']))
        label = row['toxic']
        
        # Only add non-empty text
        if text.strip():
            texts.append(text)
            labels.append(label)
    
    return texts, labels

def train_model():
    # Load and prepare data
    df = load_data()
    texts, labels = prepare_training_data(df)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # Initialize model and tokenizer
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    model = CustomToxicClassifier(base_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, augment=True)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Optimizer settings
    optimizer = AdamW([
        {'params': model.base_model.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    num_training_steps = len(train_loader) * 3  # Increase training epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_accuracy = 0
    train_losses = []
    val_accuracies = []
    patience = 3  # Early stopping patience
    no_improve = 0
    
    for epoch in range(3):  # Increase training epochs
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_losses.append(loss.item())
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        val_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                scores = torch.softmax(logits, dim=1)[:, 1]
                predictions = torch.argmax(logits, dim=-1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
                val_scores.extend(scores.cpu().numpy())
        
        accuracy = accuracy_score(val_true_labels, val_predictions)
        val_accuracies.append(accuracy)
        print(f'Validation Accuracy: {accuracy:.4f}')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improve = 0
            os.makedirs('model', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'accuracy': accuracy
            }, 'model/model.pt')
            print(f'Model saved with accuracy: {accuracy:.4f}')
            
            # Generate visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'visualizations/{timestamp}'
            
            # Plot confusion matrix
            plot_confusion_matrix(val_true_labels, val_predictions, save_path)
            
            # Plot ROC curve
            plot_roc_curve(val_true_labels, val_scores, save_path)
            
            # Plot precision-recall curve
            plot_precision_recall_curve(val_true_labels, val_scores, save_path)
            
            # Plot training metrics
            plot_training_metrics(train_losses, val_accuracies, save_path)
            
            # Save classification report
            with open(f'{save_path}/classification_report.txt', 'w') as f:
                f.write(classification_report(val_true_labels, val_predictions))
        else:
            no_improve += 1
            
        # Early stopping
        if no_improve >= patience:
            print("Early stopping triggered")
            break
        
        if accuracy >= 0.90:
            print("Target accuracy reached!")
            break
    
    print("Training completed successfully!")

if __name__ == '__main__':
    train_model() 