import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Brain, Clapperboard, Database, Film, Loader2, Search, Sparkles, TrendingUp, SlidersHorizontal } from 'lucide-react';
import './styles.css';

type MovieResult = {
  title: string;
  genres?: string;
  summary: string;
  release_year?: string;
  vote_average?: number;
  popularity?: number;
  similarity: number;
  hybrid_score?: number;
};

type SearchResponse = {
  results: MovieResult[];
  count: number;
  model: string;
  dataset: string;
  movies_loaded: number;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5001';

const sampleQueries = [
  'A space adventure with astronauts saving humanity',
  'A hacker discovers a hidden digital world and fights a powerful system',
  'A detective investigates a serial killer in a dark city',
  'A romantic story about two people from different social classes',
  'A young hero learns courage and returns to protect the kingdom',
  'A family tries to survive against terrifying creatures'
];

function scoreLabel(score: number) {
  if (score >= 0.60) return 'Excellent match';
  if (score >= 0.42) return 'Strong match';
  if (score >= 0.30) return 'Related';
  return 'Weak match';
}

function App() {
  const [query, setQuery] = useState(sampleQueries[0]);
  const [topK, setTopK] = useState(5);
  const [minScore, setMinScore] = useState(0.30);
  const [genre, setGenre] = useState('');
  const [results, setResults] = useState<MovieResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [metadata, setMetadata] = useState({ dataset: 'Not loaded yet', model: 'SBERT', movies_loaded: 0 });

  const bestScore = useMemo(() => results.length ? Math.max(...results.map((r) => r.similarity)) : 0, [results]);

  useEffect(() => {
    async function loadHealth() {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) return;
        const data = await response.json();
        setMetadata({
          dataset: data.dataset || 'Unknown dataset',
          model: data.model || 'sbert',
          movies_loaded: data.movies_loaded || 0
        });
      } catch {
        // Backend may not be running yet; search action will show the user-facing error.
      }
    }
    loadHealth();
  }, []);

  async function runSearch(nextQuery = query) {
    if (!nextQuery.trim()) {
      setError('Enter a natural-language movie description first.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: nextQuery, top_k: topK, min_score: minScore, genre })
      });
      const data: SearchResponse & { error?: string } = await response.json();
      if (!response.ok) throw new Error(data.error || 'Search failed');
      setResults(data.results || []);
      setMetadata({ dataset: data.dataset || 'Unknown dataset', model: data.model || 'sbert', movies_loaded: data.movies_loaded || 0 });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to connect to backend API. Check Flask on port 5001.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <nav className="topbar">
          <div className="brand"><Clapperboard size={26} /> CineSemantic AI</div>
          <div className="top-pills">
            <div className="pill"><Brain size={16} /> {metadata.model.toUpperCase()} + hybrid ranking</div>
            <div className="pill"><Database size={16} /> {metadata.movies_loaded ? `${metadata.movies_loaded.toLocaleString()} movies` : 'Dataset ready'}</div>
          </div>
        </nav>

        <div className="hero-grid">
          <div>
            <p className="eyebrow"><Sparkles size={16} /> Full-stack semantic recommender</p>
            <h1>Find movies by meaning, not just keywords.</h1>
            <p className="subtitle">The backend converts plot queries into embeddings, ranks movies using cosine similarity, then improves results with rating and popularity signals.</p>

            <div className="search-panel">
              <textarea value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Describe a movie plot..." />
              <div className="actions-row">
                <label>Results
                  <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                    <option value={3}>3</option><option value={5}>5</option><option value={8}>8</option><option value={10}>10</option>
                  </select>
                </label>
                <label>Genre
                  <input value={genre} onChange={(e) => setGenre(e.target.value)} placeholder="optional" />
                </label>
                <label className="slider-label"><SlidersHorizontal size={16}/> Min score {minScore.toFixed(2)}
                  <input type="range" min="0" max="0.6" step="0.02" value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} />
                </label>
                <button onClick={() => runSearch()} disabled={loading}>{loading ? <Loader2 className="spin" size={18} /> : <Search size={18} />} Search Movies</button>
              </div>
              {error && <div className="error-box">{error}</div>}
            </div>

            <div className="sample-wrap">
              {sampleQueries.map((item) => <button className="sample-chip" key={item} onClick={() => { setQuery(item); runSearch(item); }}>{item}</button>)}
            </div>
          </div>

          <aside className="insight-card">
            <TrendingUp size={28} />
            <h3>Model Signal</h3>
            <div className="metric"><span>Best Similarity</span><strong>{results.length ? bestScore.toFixed(3) : '—'}</strong></div>
            <div className="metric"><span>Returned Movies</span><strong>{results.length || '—'}</strong></div>
            <div className="metric"><span>Dataset</span><strong className="small-metric">{metadata.dataset.includes('Kaggle') ? 'Kaggle/TMDB' : 'Sample'}</strong></div>
            <div className="pipeline"><span>Query</span><span>→</span><span>SBERT</span><span>→</span><span>Hybrid Rank</span></div>
          </aside>
        </div>
      </section>

      <section className="results-layout">
        <div className="results-grid">
          {loading && Array.from({ length: topK }).map((_, i) => <div className="skeleton" key={i} />)}
          {!loading && results.map((movie, index) => (
            <article className="movie-card" key={`${movie.title}-${index}`}>
              <div className="poster-mark"><Film size={30} /></div>
              <div className="movie-body">
                <div className="rank">#{index + 1}</div>
                <h2>{movie.title} {movie.release_year && <span>({movie.release_year})</span>}</h2>
                <p className="genres">{movie.genres || 'Semantic match'}</p>
                <p className="summary">{movie.summary}</p>
                <div className="score-row"><span>{scoreLabel(movie.similarity)}</span><strong>{movie.similarity.toFixed(3)}</strong></div>
                <div className="score-bar"><span style={{ width: `${Math.min(movie.similarity * 100, 100)}%` }} /></div>
                <div className="detail-meta">Rating: {movie.vote_average || 'N/A'} · Popularity: {movie.popularity?.toFixed?.(1) || 'N/A'} · Hybrid: {movie.hybrid_score?.toFixed?.(3) || 'N/A'}</div>
              </div>
            </article>
          ))}
          {!loading && !results.length && <div className="empty-state">No results yet. Try a query or reduce the minimum score.</div>}
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
