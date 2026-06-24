from app.data_loader import load_movies
from app.evaluator import compare_models

TEST_CASES = [
    {"query": "space adventure with astronauts saving humanity", "relevant_titles": ["Interstellar", "The Martian", "Gravity", "Dune"]},
    {"query": "hacker discovers a simulated digital world", "relevant_titles": ["The Matrix"]},
    {"query": "young lion returns to reclaim his kingdom", "relevant_titles": ["The Lion King"]},
    {"query": "detectives hunt a serial killer in a dark city", "relevant_titles": ["Se7en", "Zodiac", "The Silence of the Lambs"]},
    {"query": "romance between people from different social classes", "relevant_titles": ["Titanic", "The Notebook", "Pride and Prejudice"]},
]

if __name__ == "__main__":
    movies = load_movies()
    report = compare_models(movies, TEST_CASES, k=5, model_types=("sbert",))
    print(report)
    print("\nNote: USE comparison is supported in code. Install tensorflow/tensorflow-hub and add 'use' to model_types to run it.")
