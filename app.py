"""Streamlit UI for the movie review sentiment analyzer.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.parse
import os

import pandas as pd
import streamlit as st

from src.model import DEFAULT_MODEL_PATH, top_features
from src.predict import SentimentAnalyzer

REPORT_DIR = Path(__file__).resolve().parent / "reports"
HISTORY_FILE = Path(__file__).resolve().parent / "history.json"

st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Secure TMDB API Key Loader
# ---------------------------------------------------------
def load_tmdb_api_key() -> str:
    try:
        if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass

    env_key = os.getenv("TMDB_API_KEY")
    if env_key:
        return env_key

    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return "62388b8f2e13409e4ad6eb64444a94e8"

TMDB_API_KEY = load_tmdb_api_key()

# ---------------------------------------------------------
# Persistent History Loader & Saver
# ---------------------------------------------------------
def load_persistent_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_persistent_history(history_list: list[dict]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ---------------------------------------------------------
# Live TMDB Search Helper
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def search_tmdb(query: str) -> list[dict]:
    if not query or not query.strip():
        return []
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(query.strip())}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = []
            for m in data.get("results", [])[:8]:
                year = m.get("release_date", "").split("-")[0] if m.get("release_date") else ""
                poster_path = m.get("poster_path")
                poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else None
                results.append({
                    "title": m.get("title", ""),
                    "year": f"({year})" if year else "",
                    "poster_url": poster_url
                })
            return results
    except Exception:
        pass

    q = query.lower()
    presets = [
        {"title": "Spider-Man: Brand New Day", "year": "(2026)", "poster_url": "https://image.tmdb.org/t/p/w200/8c4a8kE7PjhHwE4tWOPs3L0D3U8.jpg"},
        {"title": "Spider-Man: Homecoming", "year": "(2017)", "poster_url": "https://image.tmdb.org/t/p/w200/c24sv2weTHPsmDa7jEMN0m2P3tP.jpg"},
        {"title": "Dune", "year": "(2021)", "poster_url": "https://image.tmdb.org/t/p/w200/d5N02qvJTozScbYoWmwWF2yftMt.jpg"},
        {"title": "Guddu", "year": "(1995)", "poster_url": None}
    ]
    return [m for m in presets if q in m["title"].lower()]

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = load_persistent_history()

if "selected_movie_data" not in st.session_state:
    st.session_state["selected_movie_data"] = None

if "review_text_input" not in st.session_state:
    st.session_state["review_text_input"] = ""

if "form_msg" not in st.session_state:
    st.session_state["form_msg"] = None


@st.cache_resource
def get_analyzer() -> SentimentAnalyzer | None:
    try:
        return SentimentAnalyzer()
    except Exception:
        return None


analyzer = get_analyzer()

# ---------------------------------------------------------
# Pure Streamlit Callbacks (Executed BEFORE widget re-render)
# ---------------------------------------------------------
def cb_set_sample_review(text: str):
    st.session_state["review_text_input"] = text
    st.session_state["form_msg"] = None

def cb_select_movie(movie_dict: dict):
    st.session_state["selected_movie_data"] = movie_dict

def cb_reset_form():
    st.session_state["selected_movie_data"] = None
    st.session_state["review_text_input"] = ""
    st.session_state["movie_search_val"] = ""
    st.session_state["form_msg"] = None

def cb_clear_history():
    st.session_state["history"] = []
    save_persistent_history([])

def cb_analyze_sentiment():
    review_txt = st.session_state.get("review_text_input", "").strip()
    if not review_txt:
        st.session_state["form_msg"] = ("warning", "Please write a review or click one of the quick sample responses!")
        return

    if analyzer is None:
        return

    sel_m = st.session_state.get("selected_movie_data")
    query_title = st.session_state.get("movie_search_val", "").strip()
    
    movie_title = sel_m["title"] if sel_m else (query_title or "Untitled Movie")
    movie_year = sel_m.get("year", "") if sel_m else ""
    poster_url = sel_m.get("poster_url") if sel_m else None

    res = analyzer.predict(review_txt)

    new_entry = {
        "movie": movie_title,
        "year": movie_year,
        "poster_url": poster_url,
        "sentiment": res["label"].capitalize(),
        "confidence": res["confidence"],
        "positive_score": res["positive_score"],
        "timestamp": datetime.now().strftime("%b %d, %I:%M %p"),
        "review": review_txt,
    }
    st.session_state["history"].insert(0, new_entry)
    save_persistent_history(st.session_state["history"])
    
    st.session_state["review_text_input"] = ""
    st.session_state["selected_movie_data"] = None
    st.session_state["form_msg"] = ("success", f"Analyzed review for **{movie_title}**!")

# ---------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .block-container {
        max-width: 760px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    
    .header-badge {
        width: 44px;
        height: 44px;
        background-color: #0284c7;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 12px auto;
        color: white;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    }
    
    .main-title {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
        letter-spacing: -0.025em;
    }
    
    .main-subtitle {
        text-align: center;
        font-size: 0.92rem;
        color: #64748b;
        margin-bottom: 24px;
    }

    /* History Cards */
    .history-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .history-card:hover {
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
    }
    
    .history-body {
        display: flex;
        padding: 16px 20px;
        gap: 16px;
        align-items: flex-start;
    }
    
    .poster-container {
        width: 58px;
        height: 82px;
        flex-shrink: 0;
        border-radius: 8px;
        overflow: hidden;
        background: #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    
    .poster-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .history-card:hover .poster-img,
    .poster-container:hover .poster-img {
        transform: scale(1.4);
    }
    
    .history-content {
        flex: 1;
        min-width: 0;
    }
    
    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    
    .title-group {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    
    .movie-name {
        font-weight: 700;
        font-size: 0.98rem;
        color: #0f172a;
    }
    
    .movie-year {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 400;
    }
    
    .sentiment-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        color: white;
    }
    
    .badge-positive {
        background-color: #0284c7;
    }
    
    .badge-negative {
        background-color: #ef4444;
    }
    
    .history-date {
        font-size: 0.78rem;
        color: #94a3b8;
    }
    
    .history-text {
        font-size: 0.88rem;
        color: #475569;
        margin: 4px 0 0 0;
        line-height: 1.45;
    }
    
    .progress-track {
        width: 100%;
        height: 6px;
        background-color: #e2e8f0;
    }
    
    .progress-fill {
        height: 100%;
        background-color: #0284c7;
    }
    
    .footer-note {
        text-align: center;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Top Header
# ---------------------------------------------------------
st.markdown(
    """
<div class="header-badge">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
        <line x1="7" y1="2" x2="7" y2="22"></line>
        <line x1="17" y1="2" x2="17" y2="22"></line>
        <line x1="2" y1="12" x2="22" y2="12"></line>
        <line x1="2" y1="7" x2="7" y2="7"></line>
        <line x1="2" y1="17" x2="7" y2="17"></line>
        <line x1="17" y1="17" x2="22" y2="17"></line>
        <line x1="17" y1="7" x2="22" y2="7"></line>
    </svg>
</div>
<h1 class="main-title">Movie Review Sentiment</h1>
<p class="main-subtitle">Search a real movie, write your review, and let the rule-based classifier judge the sentiment.</p>
""",
    unsafe_allow_html=True,
)

if analyzer is None:
    st.error(f"No trained model found at `{DEFAULT_MODEL_PATH}`. Run `python train.py` first.")
    st.stop()

# ---------------------------------------------------------
# 100% PURE STREAMLIT FORM CONTAINER
# ---------------------------------------------------------
with st.container(border=True):
    st.subheader("Analyze a review", anchor=False)
    st.caption("Movie titles are looked up live from TMDB as you type.")

    # 1. Search input for movie title
    movie_search_val = st.text_input(
        "Movie title",
        key="movie_search_val",
        placeholder="Search movies, e.g. Dune",
    )

    # Render movie search suggestions if typed
    if movie_search_val.strip():
        suggestions = search_tmdb(movie_search_val.strip())
        if suggestions:
            st.markdown("<p style='font-size: 0.82rem; font-weight: 600; color: #64748b; margin-top: 6px; margin-bottom: 6px;'>Select movie suggestion:</p>", unsafe_allow_html=True)
            cols = st.columns(min(len(suggestions), 4))
            for idx, m in enumerate(suggestions[:4]):
                with cols[idx]:
                    if m.get("poster_url"):
                        st.image(m["poster_url"], width=75)
                    st.button(
                        f"🎬 {m['title']} {m['year']}",
                        key=f"sugg_m_btn_{idx}",
                        on_click=cb_select_movie,
                        args=(m,),
                    )

    # Display Selected Movie Badge if active
    if st.session_state.get("selected_movie_data"):
        sm = st.session_state["selected_movie_data"]
        m_col1, m_col2 = st.columns([1, 4])
        with m_col1:
            if sm.get("poster_url"):
                st.image(sm["poster_url"], width=90)
        with m_col2:
            st.success(f"Selected Movie: **{sm['title']}** {sm.get('year', '')}")

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # 2. Predefined Sample Review Buttons
    st.markdown("**Review**")
    st.caption("Quick sample responses (click to insert):")

    sp1, sp2, sp3, sp4 = st.columns(4)
    sp1.button(
        "👍 Masterpiece",
        key="sample_sp_1",
        on_click=cb_set_sample_review,
        args=("An absolute masterpiece with incredible direction, brilliant performances, and breathtaking visuals from start to finish!",),
        use_container_width=True,
    )
    sp2.button(
        "👍 Loved it",
        key="sample_sp_2",
        on_click=cb_set_sample_review,
        args=("Loved every minute of it! The pacing was thrilling, character arcs were well written, and the soundtrack was iconic.",),
        use_container_width=True,
    )
    sp3.button(
        "👎 Disappointing",
        key="sample_sp_3",
        on_click=cb_set_sample_review,
        args=("The plot dragged on endlessly, characters were underdeveloped, and the climax felt completely disappointing.",),
        use_container_width=True,
    )
    sp4.button(
        "👎 Poor dialogue",
        key="sample_sp_4",
        on_click=cb_set_sample_review,
        args=("Not good at all. Poor dialogue, lazy storytelling, and weak performances across the entire cast.",),
        use_container_width=True,
    )

    # 3. Review Textarea
    st.text_area(
        "Review",
        placeholder="What did you think of the movie?",
        height=120,
        label_visibility="collapsed",
        key="review_text_input",
    )

    if st.session_state.get("form_msg"):
        msg_type, msg_text = st.session_state["form_msg"]
        if msg_type == "warning":
            st.warning(msg_text)
        elif msg_type == "success":
            st.success(msg_text)

    # 4. Action Buttons (Analyze & Reset)
    act_col1, act_col2, _ = st.columns([2, 1.5, 3])
    with act_col1:
        st.button(
            "✨ Analyze sentiment",
            type="primary",
            on_click=cb_analyze_sentiment,
            use_container_width=True,
        )
    with act_col2:
        st.button(
            "🔄 Reset",
            on_click=cb_reset_form,
            use_container_width=True,
        )

# ---------------------------------------------------------
# Recent History Section (Pure Streamlit Rendering)
# ---------------------------------------------------------
st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.markdown(
        """
        <div class="section-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            Recent history
        </div>
        """,
        unsafe_allow_html=True,
    )
with h_col2:
    if st.session_state["history"]:
        st.button("🗑️ Clear history", on_click=cb_clear_history, use_container_width=True)

if not st.session_state["history"]:
    st.markdown(
        """
    <div style="text-align: center; padding: 28px; color: #94a3b8; font-size: 0.88rem; background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 12px; margin: 16px 0 20px 0;">
        No reviews analyzed yet. Search a movie and submit your review above to get started!
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    for item in st.session_state["history"]:
        badge_class = "badge-positive" if item["sentiment"].lower() == "positive" else "badge-negative"
        pos_pct = int(item["positive_score"] * 100)

        poster_html = (
            f'<img src="{item["poster_url"]}" class="poster-img"/>'
            if item.get("poster_url")
            else """
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
                <line x1="7" y1="2" x2="7" y2="22"></line>
                <line x1="17" y1="2" x2="17" y2="22"></line>
                <line x1="2" y1="12" x2="22" y2="12"></line>
                <line x1="2" y1="7" x2="7" y2="7"></line>
                <line x1="17" y1="17" x2="22" y2="17"></line>
                <line x1="17" y1="7" x2="22" y2="7"></line>
            </svg>
            """
        )

        year_str = f' <span class="movie-year">{item["year"]}</span>' if item.get("year") else ""

        card_html = f"""
        <div class="history-card">
            <div class="history-body">
                <div class="poster-container">
                    {poster_html}
                </div>
                <div class="history-content">
                    <div class="history-header">
                        <div class="title-group">
                            <span class="movie-name">{item["movie"]}</span>{year_str}
                            <span class="sentiment-badge {badge_class}">{item["sentiment"]}</span>
                        </div>
                        <span class="history-date">{item["timestamp"]}</span>
                    </div>
                    <p class="history-text">{item["review"]}</p>
                </div>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: {pos_pct}%;"></div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

st.markdown(
    """
<div class="footer-note">
    Movie data and posters provided by TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.
</div>
""",
    unsafe_allow_html=True,
)
