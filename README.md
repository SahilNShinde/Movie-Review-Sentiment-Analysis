# Movie Review Sentiment Analysis

Text-classification pipeline for movie reviews, built with scikit-learn + NLTK and trained on the
IMDB 50k dataset (25k train / 25k test). Includes training, evaluation, a CLI, and a Streamlit UI.

## Results (measured on the 25k IMDB test set)

| Algorithm | Accuracy | ROC-AUC |
|---|---|---|
| Multinomial Naive Bayes | 87.0% | 0.939 |
| Logistic Regression | 89.9% | 0.962 |
| Linear SVM (calibrated) | **90.0%** | **0.963** |

Top positive terms: great, excellent, perfect, amazing, wonderful, favorite, best, superb
Top negative terms: worst, bad, awful, boring, poor, waste, dull, terrible

Plots and full metrics are in `reports/` (`metrics.json`, `confusion_matrix.png`, `roc_curve.png`,
`top_features.json`, `comparison.json`). The trained best model ships in `models/sentiment_model.joblib`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py                    # logistic regression (default)
python train.py --algorithm nb     # multinomial naive bayes
python train.py --algorithm svm    # linear SVM
python train.py --compare          # train all three, keep the best
python train.py --sample 4000      # quick smoke run
python train.py --csv my.csv       # custom CSV with review,label columns
```

The IMDB archive (~80MB) downloads to `data/` on first run and is cached as CSV afterwards.

## CLI

```bash
python cli.py "An absolute masterpiece, I loved every minute."
python cli.py --file reviews.txt
python cli.py --csv reviews.csv --column review --out scored.csv
python cli.py --interactive
```

Output example:

```
POSITIVE  conf 97.4%  [#######################.] pos=0.974  | An absolute masterpiece...
NEGATIVE  conf 100.0% [........................] pos=0.000  | Boring, predictable and a total waste...
```

## Streamlit UI

```bash
streamlit run app.py
```

Three tabs: single-review scoring with a positivity gauge, batch CSV scoring with a sentiment chart
and download button, and a model report tab showing accuracy, ROC-AUC, per-class metrics, the
confusion matrix / ROC plots, and the most influential positive and negative terms.

## Technique

1. **Preprocessing (`src/preprocess.py`)** — lowercase, strip HTML tags, expand `n't` to `not`,
   remove punctuation, drop English stopwords *except* negation words (not/no/never/but...) so
   negated sentiment survives, then WordNet lemmatisation.
2. **Features (`src/model.py`)** — TF-IDF over unigrams + bigrams, `min_df=2`, sublinear term
   frequency, up to 100k features.
3. **Classifiers** — MultinomialNB (`alpha=0.5`), LogisticRegression (`C=8`, liblinear), and
   LinearSVC wrapped in `CalibratedClassifierCV` so it yields probabilities.
4. **Evaluation (`src/evaluate.py`)** — accuracy, precision/recall/F1 per class, ROC-AUC,
   confusion matrix and ROC curve plots.
5. **Inference (`src/predict.py`)** — `SentimentAnalyzer` loads the joblib pipeline and returns
   label, confidence, and a continuous positivity score.

## Layout

```
train.py            training + comparison entrypoint
cli.py              command-line scorer
app.py              Streamlit UI
src/data.py         IMDB download / caching / custom CSV loading
src/preprocess.py   NLTK cleaning pipeline
src/model.py        TF-IDF + classifier pipelines, save/load, top features
src/evaluate.py     metrics and plots
src/predict.py      SentimentAnalyzer inference wrapper
models/             trained pipeline (.joblib)
reports/            metrics.json and plots
```

## Measured accuracy (IMDB 50k, included model)

| Split | Size | Accuracy |
|-------|------|----------|
| Train | 25,000 | 98.59% |
| Test  | 25,000 | 90.00% |

Test-set comparison: Calibrated LinearSVM 90.00% (shipped), Logistic Regression 89.90%, Multinomial NB 86.99%.

## Quick start

```bash
unzip movie-sentiment-project.zip && cd movie-sentiment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py        # trained model included — no retraining needed
```
