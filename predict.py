# predict.py

import re
import torch
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification
from train import CustomToxicClassifier
from torch.optim import AdamW

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    # Keep apostrophes for contractions
    text = re.sub(r'[^a-z0-9\s\']', '', text)
    # Remove extra spaces but keep single spaces
    text = ' '.join(text.split())
    return text

def normalize_obfuscations(text: str) -> str:
    # First clean the text
    text = clean_text(text)
    
    # Common obfuscation patterns
    obfuscation_map = {
        r'f\*+k': 'fuck',
        r's\*+t': 'shit',
        r'a\*+s': 'ass',
        r'b\*+h': 'bitch',
        r'd\*+k': 'dick',
        r'p\*+s': 'piss',
        r'c\*+t': 'cunt',
        r'n\*+r': 'nigger',
        r'n\*+a': 'nigga',
        r'f\*+g': 'fag',
        r'f\*+t': 'fag',
        r'w\*+e': 'whore',
        r's\*+x': 'sex',
        r'p\*+n': 'porn',
        r'p\*+y': 'pussy',
        r'd\*+m': 'damn',
        r'h\*+l': 'hell'
    }
    
    # Apply obfuscation patterns
    for pattern, replacement in obfuscation_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Split into words
    words = text.split()
    normalized_words = []
    
    # Function to check if a string is likely part of a spaced-out word
    def is_word_part(s):
        # Check if the string is very short (1-2 chars) and contains only letters
        return len(s) <= 2 and s.isalpha()
    
    # Process words in groups to handle spaced-out words
    i = 0
    while i < len(words):
        # Look ahead to find potential spaced-out word
        spaced_chars = []
        j = i
        while j < len(words) and is_word_part(words[j]):
            spaced_chars.append(words[j])
            j += 1
        
        if len(spaced_chars) > 1:
            # Found a spaced-out word, combine it
            normalized_words.append(''.join(spaced_chars))
            i = j
        else:
            # Normal word, keep as is
            normalized_words.append(words[i])
            i += 1
    
    # Join all words with spaces
    text = ' '.join(normalized_words)
    
    # Apply leet speak normalization
    leet_map = str.maketrans({
        '0':'o','1':'i','3':'e','4':'a','5':'s','7':'t',
        '@':'a','$':'s','!':'i','#':'h','*':'u'
    })
    return text.translate(leet_map)

def main():
    # 1) load tokenizer & base model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    seq_model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )

    # 2) wrap & load checkpoint
    model = CustomToxicClassifier(seq_model)
    ckpt = torch.load('model/model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3) calibrated threshold and margin
    THRESHOLD = 0.452   # from calibrate_threshold.py
    MARGIN    = 0.15    # require at least 15% gap

    # 4) Context words that indicate intent
    POSITIVE_CONTEXT = {
        'genius', 'brilliant', 'amazing', 'awesome', 'great', 'excellent',
        'perfect', 'wonderful', 'fantastic', 'outstanding', 'incredible',
        'love', 'like', 'good', 'best', 'nice', 'cool', 'sweet', 'thanks',
        'thank', 'appreciate', 'helpful', 'smart', 'clever', 'talented'
    }
    
    NEGATIVE_CONTEXT = {
        'idiot', 'stupid', 'dumb', 'fool', 'moron', 'asshole', 'jerk',
        'bastard', 'bitch', 'cunt', 'dick', 'fuck', 'shit', 'piss',
        'hate', 'bad', 'worst', 'terrible', 'awful', 'horrible'
    }

    while True:
        raw = input("Enter a sentence (or 'quit' to exit): ")
        if raw.strip().lower() == 'quit':
            break

        # a) clean & normalize
        cleaned = clean_text(raw)
        normed  = normalize_obfuscations(cleaned)

        # b) tokenize & infer
        enc = tokenizer(
            normed,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        with torch.no_grad():
            logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            probs  = torch.softmax(logits, dim=1).squeeze().tolist()

        p_non = probs[0]
        p_tox = probs[1]

        # c) Check for context words
        words = set(normed.lower().split())
        has_positive = bool(words & POSITIVE_CONTEXT)
        has_negative = bool(words & NEGATIVE_CONTEXT)

        # d) margin‐based decision with context consideration
        if has_positive and has_negative:
            # Mixed context - use probability thresholds
            if p_tox > 0.6:  # High toxicity probability
                label = 'TOXIC'
                confidence = 'High' if p_tox > 0.8 else 'Medium'
            else:
                label = 'NON-TOXIC'
                confidence = 'High' if p_non > 0.8 else 'Medium'
        elif has_positive:
            # Only positive context
            label = 'NON-TOXIC'
            confidence = 'High' if p_non > 0.6 else 'Medium'
        elif has_negative:
            # Only negative context
            label = 'TOXIC'
            confidence = 'High' if p_tox > 0.6 else 'Medium'
        else:
            # No clear context - use standard thresholds
            if (p_tox > THRESHOLD) and (p_tox - p_non > MARGIN):
                label = 'TOXIC'
                confidence = 'High' if p_tox > 0.8 else 'Medium'
            else:
                label = 'NON-TOXIC'
                confidence = 'High' if p_non > 0.8 else 'Medium'

        # e) output with detailed analysis
        print(f"\nInput:             {raw}")
        print(f"Normalized Text:   {normed}")
        print(f"P(non-toxic):      {p_non:.2%}")
        print(f"P(toxic):          {p_tox:.2%}")
        print(f"Diff (tox–non):    {(p_tox-p_non):.2%}")
        print(f"Prediction:        {label}")
        print(f"Confidence:        {confidence}")
        if has_positive and has_negative:
            print(f"Note:             Detected both positive and negative context words.")
            print(f"                  The prediction is based on the overall toxicity probability.\n")
        elif has_positive:
            print(f"Note:             Detected positive context words.")
            print(f"                  The strong language appears to be used in a positive way.\n")
        elif has_negative:
            print(f"Note:             Detected negative context words.")
            print(f"                  The language appears to be used in a negative way.\n")
        elif confidence == 'Medium':
            print(f"Note:             The model is somewhat uncertain about this prediction.")
            print(f"                  Consider reviewing the context and intent.\n")
        else:
            print()

if __name__ == '__main__':
    main()
