"""Inference helpers used by the CLI and the Streamlit app."""
from __future__ import annotations

from pathlib import Path

from .model import DEFAULT_MODEL_PATH, load_model
from .preprocess import clean_text

LABELS = {0: "negative", 1: "positive"}


class SentimentAnalyzer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.pipeline = load_model(model_path)

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        pred = int(self.pipeline.predict([cleaned])[0])
        try:
            proba = self.pipeline.predict_proba([cleaned])[0]
            confidence = float(proba[pred])
            positive_score = float(proba[1])
        except AttributeError:
            margin = float(self.pipeline.decision_function([cleaned])[0])
            positive_score = 1 / (1 + pow(2.718281828, -margin))
            confidence = positive_score if pred == 1 else 1 - positive_score
        return {
            "text": text,
            "cleaned": cleaned,
            "label": LABELS[pred],
            "confidence": confidence,
            "positive_score": positive_score,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]
