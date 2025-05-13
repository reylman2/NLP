# Toxic Text Classification

This project uses DistilBERT to classify toxic text. The model is trained on the Toxic Comments dataset from Kaggle.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the model and data files:

   - Model file: [model.pt](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)
   - Training data: [toxic_comments.csv](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)
   - Test data: [test.csv](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

3. Place the files in the following structure:

```
.
├── model/
│   └── model.pt
├── data/
│   ├── toxic_comments.csv
│   └── test.csv
├── predict.py
└── requirements.txt
```

## Usage

Run the prediction script:

```bash
python predict.py
```

Enter text when prompted, and the model will classify it as toxic or non-toxic.

## Model Details

- Base model: DistilBERT
- Training data: Jigsaw Toxic Comments Classification Challenge
- Performance metrics:
  - Accuracy: XX%
  - F1 Score: XX
  - ROC AUC: XX

## License

This project is licensed under the MIT License - see the LICENSE file for details.
