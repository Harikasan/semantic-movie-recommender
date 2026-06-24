import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from app.config import (
    SBERT_MODEL_NAME,
    USE_MODEL_URL,
    DEFAULT_MIN_SCORE,
    HYBRID_SEMANTIC_WEIGHT,
    HYBRID_POPULARITY_WEIGHT,
    HYBRID_RATING_WEIGHT,
    MODELS_DIR,
)


class SemanticMovieRecommender:
    """Semantic + metadata-aware movie recommender.

    Supports SBERT by default and optional Universal Sentence Encoder for comparison.
    Ranking uses semantic similarity plus small rating/popularity boosts.
    Embeddings are cached to disk for faster repeat startup.
    """

    def __init__(self, movies_df, model_type="sbert", use_cache=True):
        self.movies_df = movies_df.reset_index(drop=True).copy()
        self.model_type = model_type.lower()
        self.use_cache = use_cache
        self.model = self._load_model()
        self.text_corpus = self.movies_df.apply(self._build_document, axis=1).tolist()
        self.embeddings = self._load_or_build_embeddings()
        self.popularity_norm = self._normalize(self.movies_df["popularity"].to_numpy(dtype=float))
        self.rating_norm = self._normalize(self.movies_df["vote_average"].to_numpy(dtype=float))

    @staticmethod
    def _normalize(values):
        values = np.nan_to_num(values, nan=0.0)
        if values.max() == values.min():
            return np.zeros_like(values, dtype=float)
        return (values - values.min()) / (values.max() - values.min())

    @staticmethod
    def _build_document(row):
        return " ".join(
            [
                str(row.get("title", "")),
                str(row.get("genres", "")),
                str(row.get("summary", "")),
            ]
        ).strip()

    def _dataset_fingerprint(self):
        """Create a stable cache key for the model and current movie corpus."""
        hasher = hashlib.blake2b(digest_size=8)
        model_name = SBERT_MODEL_NAME if self.model_type == "sbert" else USE_MODEL_URL
        hasher.update(self.model_type.encode("utf-8"))
        hasher.update(model_name.encode("utf-8"))
        hasher.update(str(len(self.text_corpus)).encode("utf-8"))
        for text in self.text_corpus:
            hasher.update(text[:500].encode("utf-8", errors="ignore"))
        return hasher.hexdigest()

    def _cache_path(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR / f"{self.model_type}_embeddings_{self._dataset_fingerprint()}.npy"

    def _load_or_build_embeddings(self):
        cache_path = self._cache_path()
        if self.use_cache and cache_path.exists():
            return np.load(cache_path)
        embeddings = self._encode(self.text_corpus)
        if self.use_cache:
            np.save(cache_path, embeddings)
        return embeddings

    def _load_model(self):
        if self.model_type == "sbert":
            return SentenceTransformer(SBERT_MODEL_NAME)
        if self.model_type == "use":
            import tensorflow_hub as hub
            return hub.load(USE_MODEL_URL)
        raise ValueError("model_type must be 'sbert' or 'use'")

    def _encode(self, texts):
        if self.model_type == "sbert":
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.array(self.model(texts))

    def search(self, query, top_k=5, min_score=DEFAULT_MIN_SCORE, genres=None):
        query_embedding = self._encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]
        hybrid_scores = (
            HYBRID_SEMANTIC_WEIGHT * semantic_scores
            + HYBRID_POPULARITY_WEIGHT * self.popularity_norm
            + HYBRID_RATING_WEIGHT * self.rating_norm
        )

        indices = np.argsort(hybrid_scores)[::-1]
        results = []
        genre_filter = genres.lower().strip() if genres else ""

        for i in indices:
            row = self.movies_df.iloc[i]
            if genre_filter and genre_filter not in str(row.get("genres", "")).lower():
                continue
            if float(semantic_scores[i]) < float(min_score):
                continue
            results.append(
                {
                    "title": row["title"],
                    "genres": row.get("genres", ""),
                    "summary": row["summary"],
                    "release_year": str(row.get("release_year", "")),
                    "vote_average": float(row.get("vote_average", 0.0)),
                    "popularity": float(row.get("popularity", 0.0)),
                    "similarity": float(semantic_scores[i]),
                    "hybrid_score": float(hybrid_scores[i]),
                }
            )
            if len(results) >= int(top_k):
                break
        return results

    def similar_movies(self, title, top_k=5, min_score=0.15):
        matches = self.movies_df[self.movies_df["title"].str.lower() == title.lower()]
        if matches.empty:
            raise ValueError(f"Movie not found: {title}")
        idx = matches.index[0]
        query_text = self.text_corpus[idx]
        results = self.search(query_text, top_k=top_k + 1, min_score=min_score)
        return [r for r in results if r["title"].lower() != title.lower()][:top_k]

    def genres(self):
        values = set()
        for genre_text in self.movies_df["genres"].fillna("").astype(str):
            for genre in genre_text.replace(",", " ").split():
                if genre and len(genre) > 2:
                    values.add(genre)
        return sorted(values)
