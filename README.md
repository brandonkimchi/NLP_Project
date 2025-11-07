To set up the project:
.\setup_env.ps1


# 🎬 Text to Rating: Sentiment Classification and Rating Approximation in Movie Reviews

**Done By:** Kanapathi Vasudevan (s3751120) & Kim E-Shawn Brandon (s3747883)  
**University of Twente – Project Group 6**

---

## 📘 Overview

This repository contains a **multi-stage NLP pipeline** for **sentiment classification** and **rating approximation** of movie reviews.  
We integrate **classical ML**, **ensemble methods**, **deep learning**, and **unsupervised clustering** to predict review polarity and estimate numerical ratings from raw text.  
The project is based on the **IMDB movie review dataset** and demonstrates how hybrid NLP systems can generate interpretable and scalable sentiment ratings.

---

## 🧩 Key Features

- 🔤 **Text Cleaning Pipeline:** Unicode normalization, slang expansion, emoji conversion, lemmatization, stopword removal  
- 🧮 **Classical Models:** TF-IDF & Bag-of-Words with Logistic Regression and Naive Bayes  
- 🧠 **Ensemble Models:** Bagging, Boosting (AdaBoost / XGBoost), Hard & Soft Voting, and Improved Stacking  
- 🤖 **Deep Learning Models:** CNN, Bi-LSTM, GloVe-LSTM, and fine-tuned **DistilBERT** transformer  
- 📊 **Unsupervised Clustering:** K-Means, Agglomerative, and DBSCAN over TF-IDF, GloVe, and Sentence-BERT embeddings  
- ⭐ **Rating Approximation:** Combines supervised sentiment probabilities with cluster-based weighting (macro, silhouette, uncertainty)  

---

## 🧠 Methodology

### 1. Dataset
**Source:** [IMDB Movie Review Dataset – Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- 50,000 labeled reviews (25k positive / 25k negative)  
- Diverse text length and tone  

### 2. Preprocessing
- HTML tag removal, Unicode normalization  
- Emoji → Text (`emoji.demojize`)  
- Slang expansion via curated dictionary  
- Tokenization + Lemmatization (spaCy, WordNet)  
- Stopword filtering and duplicate removal  

### 3. Model Training
| Category | Algorithms | Notes |
|-----------|-------------|-------|
| **Classical** | Logistic Regression, Naive Bayes | BoW / TF-IDF features |
| **Ensemble** | Random Forest, AdaBoost, XGBoost, Voting, Stacking | Meta-learning with TF-IDF features |
| **Deep Learning** | CNN, Bi-LSTM, GloVe + LSTM, DistilBERT | Contextual embeddings for semantic modeling |

Training split: **72% train / 8% validation / 20% test**

### 4. Evaluation
- Metrics: **Accuracy** & **Macro-F1**
- Tools: scikit-learn, TensorFlow/Keras, HuggingFace Transformers
- All experiments reproducible (`joblib` serialization, fixed seeds)

---

## 📊 Results Summary

| Model | Type | Accuracy |
|-------|------|-----------|
| TF-IDF + Logistic Regression | Classical | **0.8953** |
| Improved Stacking Ensemble | Ensemble | **0.907** |
| DistilBERT (Fine-Tuned) | Transformer | **0.9132** |

### 🧮 Rating Approximation 

| Method | Score (/10) |
|---------|-------------|
| Supervised Micro Average | 7.87 |
| Unsupervised Macro-by-Cluster | 7.87 |
| Uncertainty-Weighted | 8.25 |
| Silhouette-Weighted | 7.85 |

> The hybrid supervised + unsupervised method captures **sentiment strength distribution** rather than a simple average polarity.

---

## 🧩 Insights

- 🏆 **DistilBERT** achieved the best accuracy overall.  
- ⚖️ **Improved stacking** nearly matched transformer performance with better interpretability.  
- 💡 The pipeline generalizes well to informal datasets (e.g., YouTube comments).

---

## ⚙️ Tech Stack

- **Languages:** Python  
- **Libraries:** scikit-learn, TensorFlow, Keras, HuggingFace Transformers, Gensim, SpaCy, NLTK, XGBoost  
- **Utilities:** `joblib`, `emoji`, `matplotlib`, `pandas`, `numpy`  

---

## 🧭 Repository Structure

