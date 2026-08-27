"""Streamlit UI for the movie review sentiment analyzer.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import DEFAULT_MODEL_PATH, top_features
from src.predict import SentimentAnalyzer

REPORT_DIR = Path(__file__).resolve().parent / "reports"

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="*", layout="wide")


@st.cache_resource
def get_analyzer() -> SentimentAnalyzer | None:
    try:
        return SentimentAnalyzer()
    except FileNotFoundError:
        return None


st.title("Movie Review Sentiment Analyzer")
st.caption("TF-IDF text classification trained on the IMDB 50k review dataset")

analyzer = get_analyzer()
if analyzer is None:
    st.error(f"No trained model found at `{DEFAULT_MODEL_PATH}`. Run `python train.py` first.")
    st.stop()

single, batch, metrics_tab = st.tabs(["Single review", "Batch (CSV)", "Model report"])

with single:
    text = st.text_area(
        "Paste a movie review",
        height=180,
        placeholder="The pacing dragged in the second act, but the lead performance was extraordinary...",
    )
    if st.button("Analyze", type="primary") and text.strip():
        res = analyzer.predict(text)
        col1, col2 = st.columns([1, 2])
        with col1:
            if res["label"] == "positive":
                st.success(f"Positive ({res['confidence']:.1%} confidence)")
            else:
                st.error(f"Negative ({res['confidence']:.1%} confidence)")
        with col2:
            st.progress(res["positive_score"], text=f"Positivity score {res['positive_score']:.3f}")
        with st.expander("Preprocessed text"):
            st.code(res["cleaned"] or "(empty after cleaning)")

with batch:
    uploaded = st.file_uploader("CSV file with a review column", type=["csv"])
    column = st.text_input("Column name", value="review")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if column not in df.columns:
            st.warning(f"Column '{column}' not in CSV. Found: {', '.join(df.columns)}")
        else:
            results = analyzer.predict_batch(df[column].astype(str).tolist())
            out = df.copy()
            out["sentiment"] = [r["label"] for r in results]
            out["confidence"] = [round(r["confidence"], 4) for r in results]
            out["positive_score"] = [round(r["positive_score"], 4) for r in results]
            st.dataframe(out, use_container_width=True)
            counts = out["sentiment"].value_counts()
            st.bar_chart(counts)
            st.download_button(
                "Download scored CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="scored_reviews.csv",
                mime="text/csv",
            )

with metrics_tab:
    mfile = REPORT_DIR / "metrics.json"
    if mfile.exists():
        m = json.loads(mfile.read_text())
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        st.dataframe(pd.DataFrame(m["report"]).T, use_container_width=True)
        imgs = [p for p in ("confusion_matrix.png", "roc_curve.png") if (REPORT_DIR / p).exists()]
        if imgs:
            for col, name in zip(st.columns(len(imgs)), imgs):
                col.image(str(REPORT_DIR / name))
    else:
        st.info("No metrics yet - run `python train.py`.")

    feats = top_features(analyzer.pipeline, n=15)
    if feats["positive"]:
        fc1, fc2 = st.columns(2)
        fc1.subheader("Most positive terms")
        fc1.dataframe(pd.DataFrame(feats["positive"], columns=["term", "weight"]), hide_index=True)
        fc2.subheader("Most negative terms")
        fc2.dataframe(pd.DataFrame(feats["negative"], columns=["term", "weight"]), hide_index=True)
