# src/training.py
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

def build_models():
    """Classic baselines: BoW/TF-IDF × {LogReg, ComplementNB}."""
    return {
        # Bag of Words
        "bow_lr": Pipeline([
            ("vect", CountVectorizer(max_features=50_000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
        ]),
        "bow_nb": Pipeline([
            ("vect", CountVectorizer(max_features=50_000, ngram_range=(1, 2))),
            ("clf", ComplementNB())
        ]),
        # TF-IDF
        "tfidf_lr": Pipeline([
            ("vect", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
        ]),
        "tfidf_nb": Pipeline([
            ("vect", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
            ("clf", ComplementNB())
        ]),
    }

def train_eval(name, model, X_tr, y_tr, X_va, y_va, X_te, y_te, out_dir):
    """Fit on TRAIN, report on VAL and TEST, then save."""
    print(f"\n========== {name} ==========")
    model.fit(X_tr, y_tr)

    print("\nValidation results:")
    va_pred = model.predict(X_va)
    print(classification_report(y_va, va_pred, digits=4))

    print("\nTest results:")
    te_pred = model.predict(X_te)
    print(classification_report(y_te, te_pred, digits=4))

    acc = accuracy_score(y_te, te_pred)
    f1  = f1_score(y_te, te_pred, average="macro")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved -> {path}")

    return {"model": name, "test_acc": acc, "test_f1": f1}

def main():
    data_path = "data/IMDB_clean.csv"
    out_dir = "models"

    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise KeyError(f"{data_path} must have columns: text,label")
    X, y = df["text"], df["label"].astype(int)

    # Train / Test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # Train / Val split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=42
    )

    models = build_models()
    results = []
    for name, model in models.items():
        res = train_eval(name, model, X_tr, y_tr, X_va, y_va, X_te, y_te, out_dir)
        results.append(res)

    print("\n=== Summary (Test) ===")
    for r in results:
        print(f"{r['model']:10s}  ACC={r['test_acc']:.4f}  F1={r['test_f1']:.4f}")

if __name__ == "__main__":
    main()
