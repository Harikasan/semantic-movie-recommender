from app.data_loader import load_movies
from app.visualization import generate_tsne

if __name__ == "__main__":
    movies = load_movies()
    path = generate_tsne(movies)
    print(f"Saved visualization to {path}")
