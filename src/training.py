import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
import joblib


def build_models():
    """Return a dictionary of classic model pipelines."""
    models = {
        # Bag-of-Words
        "bow_lr": Pipeline([
            ("vect", CountVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
        ]),
        "bow_nb": Pipeline([
            ("vect", CountVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", ComplementNB())
        ]),

        # TF-IDF
        "tfidf_lr": Pipeline([
            ("vect", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
        ]),
        "tfidf_nb": Pipeline([
            ("vect", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", ComplementNB())
        ]),
    }
    return models


def evaluate_model(name, model, X_val, y_val, X_test, y_test, out_dir):
    """Fit model, print metrics, and save."""
    print(f"\n{'='*10} Training {name} {'='*10}")
    model.fit(X_val, y_val)

    print("\nValidation Results:")
    val_pred = model.predict(X_val)
    print(classification_report(y_val, val_pred, digits=4))

    print("\nTest Results:")
    test_pred = model.predict(X_test)
    print(classification_report(y_test, test_pred, digits=4))

    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, average="macro")

    # Save model
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved model -> {path}")

    return {"model": name, "test_acc": acc, "test_f1": f1}


def main():
    data_path = "data/IMDB_clean.csv"
    out_dir = "models"
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    results = []
    models = build_models()

    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test, out_dir)
        results.append(res)

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['model']:10s}  ACC={r['test_acc']:.4f}  F1={r['test_f1']:.4f}")


if __name__ == "__main__":
    main()
