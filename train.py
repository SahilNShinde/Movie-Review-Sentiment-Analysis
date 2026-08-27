"""Train a movie review sentiment classifier on the IMDB 50k dataset.

Examples:
    python train.py                              # logistic regression (default)
    python train.py --algorithm nb               # multinomial naive bayes
    python train.py --algorithm svm --compare    # train all three and compare
    python train.py --csv my_reviews.csv         # custom labelled CSV
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data import load_csv, load_imdb
from src.evaluate import REPORT_DIR, evaluate
from src.model import ALGORITHMS, DEFAULT_MODEL_PATH, build_pipeline, save_model, top_features
from src.preprocess import clean_series


def get_data(csv: str | None, sample: int | None):
    if csv:
        df = load_csv(csv)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
    else:
        train_df, test_df = load_imdb()
    if sample:
        train_df = train_df.head(sample)
        test_df = test_df.head(max(sample // 4, 100))
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMDB sentiment classifier")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="logreg")
    parser.add_argument("--compare", action="store_true", help="train and compare all algorithms")
    parser.add_argument("--csv", help="path to a custom CSV (review,label)")
    parser.add_argument("--sample", type=int, help="limit rows for a quick run")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    train_df, test_df = get_data(args.csv, args.sample)
    print(f"Train: {len(train_df):,} reviews | Test: {len(test_df):,} reviews")

    print("Cleaning text (NLTK lemmatisation + stopword removal)...")
    X_train = clean_series(train_df["review"])
    X_test = clean_series(test_df["review"])
    y_train, y_test = train_df["label"].tolist(), test_df["label"].tolist()

    algorithms = list(ALGORITHMS) if args.compare else [args.algorithm]
    results: dict[str, float] = {}
    best_name, best_pipe, best_acc = None, None, -1.0

    for name in algorithms:
        print(f"\n=== {name} ===")
        started = time.time()
        pipe = build_pipeline(name)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics["accuracy"]
        print(f"trained in {time.time() - started:.1f}s")
        if metrics["accuracy"] > best_acc:
            best_name, best_pipe, best_acc = name, pipe, metrics["accuracy"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feats = top_features(best_pipe)
    (REPORT_DIR / "top_features.json").write_text(json.dumps(feats, indent=2))
    (REPORT_DIR / "comparison.json").write_text(json.dumps(results, indent=2))

    path = save_model(best_pipe, Path(args.model_path))
    print(f"\nBest model: {best_name} (accuracy {best_acc:.4f}) -> {path}")
    print("Top positive terms:", ", ".join(t for t, _ in feats["positive"][:10]))
    print("Top negative terms:", ", ".join(t for t, _ in feats["negative"][:10]))


if __name__ == "__main__":
    main()
