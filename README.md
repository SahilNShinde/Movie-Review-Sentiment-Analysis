# 🎬 Movie Review Sentiment Analysis

> 🌐 **Live Demo:** [Click here to view live web app]*((https://movie-review-sentiment-analysis-hnbcljs5nzsvqgbanpumkp.streamlit.app/))*

An interactive end-to-end Machine Learning & NLP web application that analyzes movie reviews and classifies their sentiment as **Positive** or **Negative** in real-time. Built with Python, Streamlit, and Scikit-Learn, featuring live TMDB API integration and persistent review history.

---

## 🚀 Key Features

* **🔍 Live TMDB Autocomplete Search:** Instant live movie search powered by The Movie Database (TMDB) REST API, featuring release years and poster previews.
* **💬 Quick Predefined Sample Responses:** One-click sample review buttons (both positive and negative) for fast testing without typing.
* **⚡ 100% Native Streamlit Architecture:** Built strictly with native Streamlit widgets and event callbacks (`on_click`) for instant responsiveness and zero buffering.
* **💾 Persistent History Storage:** All analyzed movie reviews are saved to disk (`history.json`) and automatically loaded whenever you open the app.
* **📊 Sentiment Classification & Confidence:** Evaluates text sentiment and renders confidence progress bars with positive/negative badges.

---

## 🛠️ Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Frontend UI** | Streamlit, Custom CSS |
| **Machine Learning / NLP** | Scikit-Learn (TF-IDF Vectorizer + Classifier), Python 3.x |
| **External API** | TMDB REST API (Movie search & poster artwork) |
| **Data Processing** | Pandas, NumPy, `urllib`, `json` |
| **Storage & Persistence** | Persistent JSON Storage (`history.json`) |

---

## 📚 Dataset & Model Metrics

The sentiment classifier is trained on the benchmark **[IMDb 50K Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)**.

| Metric | Score |
| :--- | :--- |
| **Total Samples** | 50,000 balanced reviews (25,000 Positive / 25,000 Negative) |
| **Training Accuracy** | **98.59%** |
| **Testing Accuracy** | **90.00%** |

---

## 📁 Project Structure

```text
Movie-Review-Sentiment-Analysis/
├── app.py                  # Main Streamlit web application
├── history.json            # Persistent storage for user review history
├── requirements.txt        # Python dependencies
├── src/                    # Source code & ML model pipeline
│   ├── model.py            # Model training & vectorization definitions
│   └── predict.py          # Sentiment analyzer inference class
├── reports/                # Classification reports & metrics
└── README.md               # Project documentation
```

---



## 📄 License & Attribution

Movie data and poster images provided by [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.
