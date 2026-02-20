# IT549: Deep Learning — Lab 2
### GloVe Pretrained Embeddings for Movie Text Prediction

---

Name: Khushali Mandalia  
ID: 2025110225  
Course: IT549 – Deep Learning  

---

## What This Project Does

This lab uses pretrained GloVe word embeddings (100D) to build two models on a movie dataset:

1. **Regression** – predict a movie's rating from its text (overview / tagline / keywords)
2. **Genre Classification** – predict which genres a movie belongs to (multi-label)

We also do some text analysis to find which words are most common and most indicative per genre.

---

## Dataset

[Kaggle Movie Dataset](https://www.kaggle.com/datasets/figolm10/moviedataset)  
Columns used: `overview`, `tagline`, `keywords`, `genre`, `voting_average`

---

## How It Works

- Text is cleaned (lowercase, remove punctuation/numbers, lemmatize)
- Each document is converted to a 100D vector using **TF-IDF weighted GloVe averaging** — rare, distinctive words get more weight than common ones
- A small MLP neural network is trained separately for each text column
- Data split: 70% train / 15% val / 15% test

---

## Results

### Regression (predicting rating)

| Input | MSE | RMSE |
|-------|-----|------|
| Baseline (mean) | — | — |
| overview | — | — |
| tagline | — | — |

### Genre Classification

| Input | Micro-F1 | Macro-F1 | Hamming Loss |
|-------|----------|----------|--------------|
| overview | — | — | — |
| keywords | — | — | — |

> Fill in your numbers after running the notebook.

---

## How to Run

1. Upload your CSV and `glove.6B.100d.txt` to Colab via the 📁 sidebar
2. Install dependencies: `pip install torch scikit-learn pandas numpy nltk tqdm matplotlib`
3. Run all cells top to bottom in `IT549_Lab2_GloVe_Movie_Prediction.ipynb`

---

## References
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Kaggle Movie Dataset](https://www.kaggle.com/datasets/figolm10/moviedataset)

