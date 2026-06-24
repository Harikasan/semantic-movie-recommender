import ast
from pathlib import Path
import pandas as pd
from app.config import DEFAULT_DATA_PATH, KAGGLE_MOVIES_PATH


def _parse_genres(value):
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return " ".join([str(item.get("name", "")) for item in parsed if isinstance(item, dict)]).strip()
            except Exception:
                return text
        return text
    return str(value)


def _normalize_movies(df):
    rename_map = {}
    if "overview" in df.columns and "summary" not in df.columns:
        rename_map["overview"] = "summary"
    if "movie_title" in df.columns and "title" not in df.columns:
        rename_map["movie_title"] = "title"
    df = df.rename(columns=rename_map)

    required = {"title", "summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain title and overview/summary columns. Missing: {missing}")

    keep = [c for c in ["title", "summary", "genres", "vote_average", "popularity", "release_date"] if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["title", "summary"])
    df["title"] = df["title"].astype(str).str.strip()
    df["summary"] = df["summary"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["summary"] != "")]

    if "genres" not in df.columns:
        df["genres"] = ""
    df["genres"] = df["genres"].apply(_parse_genres)

    for col in ["vote_average", "popularity"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "release_date" not in df.columns:
        df["release_date"] = ""
    df["release_year"] = df["release_date"].astype(str).str[:4].where(df["release_date"].notna(), "")

    df = df.drop_duplicates(subset=["title", "summary"]).reset_index(drop=True)
    return df


def load_movies(path=None, limit=None):
    """Load either Kaggle/TMDB movies_metadata.csv or the bundled sample file."""
    chosen = Path(path) if path else (KAGGLE_MOVIES_PATH if KAGGLE_MOVIES_PATH.exists() else DEFAULT_DATA_PATH)
    df = pd.read_csv(chosen, low_memory=False)
    df = _normalize_movies(df)
    if limit:
        df = df.head(int(limit)).copy()
    return df
