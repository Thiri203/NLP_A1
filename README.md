# A1 – Word Embedding Models (Skip-gram, NEG, GloVe)

This repository contains my implementation for **Assignment 1** of NLP.
The project covers training word embedding models, evaluation, and a simple web application.

---

## Project Structure

```
A1_ST126018/
├── app/
│   ├── main.py              # Flask web application
│   └── templates/
│       └── index.html       # Simple UI
├── artifacts/
│   ├── skipgram_model.pkl
│   ├── skipgram_neg_model.pkl
│   ├── glove_model.pkl
│   ├── word2idx.pkl
│   └── idx2word.pkl
├── a1.ipynb                 # Training + evaluation notebook
├── wordsim353.csv
├── word-test.v1.txt
├── requirements.txt
└── README.md
```

---

## Task 1 – Training Word Embeddings

I trained three models:
- **Skip-gram**
- **Skip-gram with Negative Sampling**
- **GloVe**

Early stopping was applied when loss stopped improving.

---

## Task 2 – Evaluation

### Training Time

| Model | Training Time |
|------|--------------|
| Skip-gram | 811.89 s |
| Skip-gram + NEG | 1532.75 s |
| GloVe | 70.43 s |

### Analogy Accuracy

| Model | Semantic Acc | Syntactic Acc |
|------|-------------|---------------|
| Skip-gram | 0.0000 | 0.0007 |
| Skip-gram + NEG | 0.0210 | 0.0117 |
| GloVe | 0.0000 | 0.0002 |

### WordSim353 (Spearman Correlation)

| Model | Spearman |
|------|----------|
| Skip-gram | -0.0730 |
| Skip-gram + NEG | 0.2621 |
| GloVe | 0.1367 |

Skip-gram + NEG shows the best correlation with human judgment.

---

## Task 3 – Web Application

A simple Flask web app is implemented to search similar contexts using dot-product similarity.

### How it works
1. User enters a query word
2. The query vector is compared with all word vectors
3. Top-10 most similar words are returned

---

## How to Run the Application

### 1. Create virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask app

From project root:

```bash
python app/main.py
```

### 4. Open browser

```
http://127.0.0.1:5000
```

---

## Notes
- UI is intentionally minimal
- Focus is on correctness and functionality
- All trained models are stored under `artifacts/`

---

**Student:** Thiri Shin Thant  
**Course:** NLP  
