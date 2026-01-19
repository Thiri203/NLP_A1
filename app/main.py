from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ---------- Utils ----------
def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

def cosine_sim(a, b):
    return np.dot(a, b)

# ---------- Load artifacts ----------
# GloVe
with open("../artifacts/glove_model.pkl", "rb") as f:
    data = pickle.load(f)

# 🔴 FIX: your pickle does NOT use "vectors"
# Check common cases safely
if "embeddings" in data:
    glove_embeddings = data["embeddings"]
elif "glove_embeddings" in data:
    glove_embeddings = data["glove_embeddings"]
elif "W" in data and "W_tilde" in data:
    glove_embeddings = data["W"] + data["W_tilde"]
else:
    raise KeyError("Cannot find embeddings inside glove_model.pkl")

# word mappings
with open("../artifacts/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("../artifacts/idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

# normalize embeddings
glove_embeddings = normalize(glove_embeddings)
UNK_ID = word2idx.get("<UNK>", 0)

# ---------- Similarity search ----------
def search_similar(query, top_k=10):
    words = query.lower().split()
    vectors = []

    for w in words:
        idx = word2idx.get(w, UNK_ID)
        vectors.append(glove_embeddings[idx])

    if len(vectors) == 0:
        return []

    query_vec = np.mean(vectors, axis=0)
    scores = np.dot(glove_embeddings, query_vec)

    top_idx = np.argsort(scores)[-top_k:][::-1]
    results = [(idx2word[i], float(scores[i])) for i in top_idx]
    return results

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form["query"]
        results = search_similar(query)

    return render_template("index.html", query=query, results=results)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
