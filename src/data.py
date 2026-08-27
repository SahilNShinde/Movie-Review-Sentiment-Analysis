"""IMDB 50k dataset loading utilities."""
from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def download_imdb(cache_dir: Path = CACHE_DIR) -> Path:
    """Download and extract the IMDB 50k dataset (cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / "aclImdb"
    if root.exists():
        return root
    archive = cache_dir / "aclImdb_v1.tar.gz"
    if not archive.exists():
        print("Downloading IMDB dataset (~80MB)...")
        urllib.request.urlretrieve(IMDB_URL, archive)
    print("Extracting...")
    with tarfile.open(archive) as tar:
        tar.extractall(cache_dir)
    return root


def _read_split(root: Path, split: str) -> pd.DataFrame:
    rows = []
    for label, target in (("pos", 1), ("neg", 0)):
        folder = root / split / label
        for file in folder.glob("*.txt"):
            rows.append({"review": file.read_text(encoding="utf-8"), "label": target})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def load_imdb(cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with columns: review, label (1=positive)."""
    csv_train = cache_dir / "imdb_train.csv"
    csv_test = cache_dir / "imdb_test.csv"
    if csv_train.exists() and csv_test.exists():
        return pd.read_csv(csv_train), pd.read_csv(csv_test)
    root = download_imdb(cache_dir)
    train, test = _read_split(root, "train"), _read_split(root, "test")
    train.to_csv(csv_train, index=False)
    test.to_csv(csv_test, index=False)
    return train, test


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a custom CSV with `review` and `label` columns."""
    df = pd.read_csv(path)
    missing = {"review", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df
