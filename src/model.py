"""Model factory: TF-IDF + (Multinomial Naive Bayes | Logistic Regression | LinearSVC)."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"

ALGORITHMS = ("nb", "logreg", "svm")


def build_pipeline(algorithm: str = "logreg", max_features: int = 100_000) -> Pipeline:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )

    if algorithm == "nb":
        clf = MultinomialNB(alpha=0.5)
    elif algorithm == "logreg":
        clf = LogisticRegression(C=8.0, max_iter=2000, solver="liblinear")
    else:
        clf = CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def save_model(pipeline: Pipeline, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python train.py --algorithm logreg"
        )
    return joblib.load(path)


def top_features(pipeline: Pipeline, n: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Most positive / negative tokens for linear models."""
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    coefs = getattr(clf, "coef_", None)
    if coefs is None and hasattr(clf, "calibrated_classifiers_"):
        coefs = clf.calibrated_classifiers_[0].estimator.coef_
    if coefs is None:
        if hasattr(clf, "feature_log_prob_"):
            coefs = (clf.feature_log_prob_[1] - clf.feature_log_prob_[0]).reshape(1, -1)
        else:
            return {"positive": [], "negative": []}
    weights = coefs[0]
    names = vec.get_feature_names_out()
    order = weights.argsort()
    return {
        "negative": [(names[i], float(weights[i])) for i in order[:n]],
        "positive": [(names[i], float(weights[i])) for i in order[-n:][::-1]],
    }
