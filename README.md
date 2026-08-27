# 🎬 Movie Review Sentiment Analysis

An interactive end-to-end Machine Learning web application that analyzes movie reviews and classifies their sentiment as **Positive** or **Negative** in real-time. Built with Python and Streamlit, featuring a pre-trained NLP classification model.

---

## 🚀 Features

* **Real-time Sentiment Classification:** Predicts whether a movie review is positive or negative instantly.
* **Interactive Dashboard:** Clean, intuitive user interface powered by Streamlit.
* **Pre-trained Model Integration:** The serialized trained model ships directly with the project—no retraining required on startup.
* **High Performance:** Achieves high accuracy across training and test splits.

---

## 📚 Dataset

This model is trained on the benchmark **[IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)** (commonly known as the **IMDb 50K Movie Reviews Dataset**).

* **Total Samples:** 50,000 highly polarized reviews
* **Class Distribution:** Perfectly balanced (25,000 Positive, 25,000 Negative)
* **Labels:** Binary (`1` for Positive $\ge 7/10$ rating, `0` for Negative $\le 4/10$ rating)
* **Preprocessing:** HTML tag stripping, punctuation removal, lowercasing, stopword filtering, and TF-IDF / Bag-of-Words text vectorization.

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | 98.59% |
| **Testing Accuracy** | 90.00% |

---

## 📁 Project Structure

```text
Movie-Review-Sentiment-Analysis/
├── src/                    # Source code & model training scripts
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Required Python dependencies
└── README.md               # Project documentation
