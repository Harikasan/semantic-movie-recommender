from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "sample_movies.csv"
KAGGLE_MOVIES_PATH = DATA_DIR / "movies_metadata.csv"
MODELS_DIR = BASE_DIR / "models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.20
HYBRID_SEMANTIC_WEIGHT = 0.85
HYBRID_POPULARITY_WEIGHT = 0.10
HYBRID_RATING_WEIGHT = 0.05
