from app.data_loader import load_movies
from app.recommender import SemanticMovieRecommender


def test_search_returns_results():
    movies = load_movies()
    recommender = SemanticMovieRecommender(movies)
    results = recommender.search("space adventure with astronauts", top_k=3, min_score=0.0)
    assert len(results) == 3
    assert "title" in results[0]
    assert "similarity" in results[0]


def test_similar_movies_excludes_query_title():
    movies = load_movies()
    recommender = SemanticMovieRecommender(movies)
    results = recommender.similar_movies("Interstellar", top_k=3, min_score=0.0)
    assert all(r["title"] != "Interstellar" for r in results)
