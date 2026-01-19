# NLP Assignment 1 – Word Embeddings and Similarity Search  
**Student Name:** Thiri Shin Thant  
**Student ID:** ST126018  

---

## 📌 Project Overview

This project implements and compares three word embedding models:

1. Skip-gram  
2. Skip-gram with Negative Sampling (NEG)  
3. GloVe  

The models are trained from scratch and evaluated using word similarity, word analogy tasks, training time, and a simple web application.

---

## 📂 Project Structure

```
A1_ST126018/
│
├── app/
│   ├── main.py
│   └── templates/
│       └── index.html
│
├── artifacts/
│   ├── glove_model.pkl
│   ├── skipgram_model.pkl
│   ├── skipgram_neg_model.pkl
│   ├── word2idx.pkl
│   └── idx2word.pkl
│
├── a1.ipynb
├── word-test.v1.txt
├── wordsim353.csv
├── README.md
└── requirements / config files
```

---

## 🧠 Tasks Implemented

### Task 1: Training Word Embeddings
- Skip-gram (with early stopping)
- Skip-gram + Negative Sampling
- GloVe (co-occurrence based)
- Maximum 10 epochs per model

---

### Task 2: Evaluation

#### Word Similarity (WordSim353 – Spearman Correlation)

| Model | Correlation |
|------|------------|
| Skip-gram | -0.073 |
| Skip-gram + NEG | 0.262 |
| GloVe | 0.137 |

#### Word Analogy Accuracy (word-test.v1)

| Model | Semantic | Syntactic |
|------|----------|-----------|
| Skip-gram | 0.00 | 0.0007 |
| Skip-gram + NEG | 0.021 | 0.0117 |
| GloVe | 0.00 | 0.0002 |

#### Training Time

| Model | Time (seconds) |
|------|----------------|
| Skip-gram | 811.89 |
| Skip-gram + NEG | 1532.75 |
| GloVe | 70.43 |

---

## 🌐 Task 3: Web Application

A simple Flask web application is developed to search similar words using dot-product similarity.

### How to Run

```bash
cd app
py main.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## 🛠 Tools Used

- Python
- PyTorch
- NumPy
- SciPy
- Flask
- Jupyter Notebook

---

## ✅ Conclusion

Skip-gram with Negative Sampling performs best overall in both similarity and analogy tasks.  
GloVe trains much faster but gives lower accuracy.  
The web application shows a practical use of word embeddings.

