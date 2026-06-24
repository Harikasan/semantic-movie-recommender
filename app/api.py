from flask import Flask, jsonify, request
from flask_cors import CORS
from app.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, KAGGLE_MOVIES_PATH
from app.data_loader import load_movies
from app.recommender import SemanticMovieRecommender

app = Flask(__name__)
CORS(app)

movies_df = load_movies()
dataset_name = "Kaggle/TMDB movies_metadata.csv" if KAGGLE_MOVIES_PATH.exists() else "Bundled sample dataset"
recommender = SemanticMovieRecommender(movies_df, model_type="sbert", use_cache=True)


def _int_body_value(body, key, default, minimum=1, maximum=25):
    try:
        value = int(body.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _float_body_value(body, key, default, minimum=0.0, maximum=1.0):
    try:
        value = float(body.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@app.get("/")
def root():
    return jsonify({
        "app": "CineSemantic AI",
        "status": "running",
        "frontend": "http://localhost:5173",
        "endpoints": ["GET /health", "GET /genres", "POST /search", "POST /recommend", "POST /similar"],
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "movies_loaded": len(movies_df),
        "dataset": dataset_name,
        "model": recommender.model_type,
        "cache_enabled": recommender.use_cache,
    })


@app.get("/genres")
def genres():
    return jsonify({"genres": recommender.genres()})


@app.post("/search")
def search():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", "")).strip()
    top_k = _int_body_value(body, "top_k", DEFAULT_TOP_K, minimum=1, maximum=25)
    min_score = _float_body_value(body, "min_score", DEFAULT_MIN_SCORE, minimum=0.0, maximum=1.0)
    genre = str(body.get("genre", "")).strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = recommender.search(query, top_k=top_k, min_score=min_score, genres=genre)
    return jsonify({
        "results": results,
        "count": len(results),
        "model": recommender.model_type,
        "dataset": dataset_name,
        "movies_loaded": len(movies_df),
        "min_score": min_score,
    })


@app.post("/recommend")
def recommend():
    return search()


@app.post("/similar")
def similar():
    body = request.get_json(silent=True) or {}
    title = str(body.get("title", "")).strip()
    top_k = _int_body_value(body, "top_k", DEFAULT_TOP_K, minimum=1, maximum=25)

    if not title:
        return jsonify({"error": "title is required"}), 400

    try:
        return jsonify({
            "results": recommender.similar_movies(title, top_k=top_k),
            "model": recommender.model_type,
            "dataset": dataset_name,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
