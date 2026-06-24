from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from app.config import VISUALIZATIONS_DIR
from app.recommender import SemanticMovieRecommender


def generate_tsne(movies_df, output_path=None, limit=300):
    limited = movies_df.head(limit).copy()
    recommender = SemanticMovieRecommender(limited, model_type="sbert")
    perplexity = max(2, min(30, len(limited) - 1))
    coords = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="random", learning_rate="auto").fit_transform(recommender.embeddings)

    output = Path(output_path) if output_path else VISUALIZATIONS_DIR / "movie_embeddings_tsne.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=28, alpha=0.75)
    for i, title in enumerate(limited["title"].head(40)):
        plt.annotate(title[:24], (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.85)
    plt.title("t-SNE Visualization of Movie Plot Embeddings")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return output
