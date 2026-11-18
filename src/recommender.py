"""Loads the findings exported from the notebook and ranks movies."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ART_DIR

MODES = ("Hybrid", "Semantic", "TF-IDF")
FILES = ("movies.json", "sim_tfidf.npy", "sim_semantic.npy")


def _load_movies() -> pd.DataFrame:
    # Load the JSON file we exported from the notebook
    with open(ART_DIR / "movies.json", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    # Clean up missing data so the UI doesn't crash
    df["title"] = df["title"].fillna(df["original_title"])
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    for col in ("vote_average", "vote_count", "runtime"):
        df[col] = df[col].fillna(0)
    # Create unique labels (e.g., "Avatar (2009)") for the search dropdown. 
    # Create new labels for search If two movies have the exact same name and year, it adds their TMDB ID to tell them apart.
    label = df["title"] + df["year"].map(lambda y: f" ({int(y)})" if pd.notna(y) else "")
    dup = label.duplicated(keep=False)
    df["label"] = label.where(~dup, label + " [id " + df["movie_id"].astype(str) + "]")
    return df


class Recommender:
    def __init__(self, df: pd.DataFrame, sim_tfidf: np.ndarray, sim_semantic: np.ndarray):
        # Safety check: Ensure the dataset size matches the matrices size
        assert sim_tfidf.shape == sim_semantic.shape == (len(df), len(df)), \
            "Exported matrices don't match movies.json. Re-export from the notebook."
        self.df, self.sim_tfidf, self.sim_semantic = df, sim_tfidf, sim_semantic
        # Create lookup dictionaries so we can easily find a movie's index by its title
        self.labels = df["label"].tolist()
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        # Extract unique genres for the Streamlit sidebar filter
        self.all_genres = sorted({g for gs in df["genres"] for g in gs})
        self._gsets = [set(g) for g in df["genres"]]
        self._rating = df["vote_average"].to_numpy()

    @classmethod
    def load(cls) -> "Recommender":
        # Check if the user remembered to put the zip files in the artifacts folder
        if not all((ART_DIR / f).exists() for f in FILES):
            raise FileNotFoundError(
                "Notebook exports not found in artifacts/ "
                f"(need: {', '.join(FILES)})")
        # Load the massive matrices using mmap_mode="r"
        return cls(_load_movies(),
                   np.load(ART_DIR / "sim_tfidf.npy", mmap_mode="r"),
                   np.load(ART_DIR / "sim_semantic.npy", mmap_mode="r"))

    def scores(self, i: int, mode: str = "Hybrid", alpha: float = 0.5) -> np.ndarray:
        # Fetches the similarity scores for movie 'i' based on the selected mode
        sem = np.asarray(self.sim_semantic[i], dtype=np.float32)
        tf = np.asarray(self.sim_tfidf[i], dtype=np.float32)
        if mode == "Semantic":
            return sem
        if mode == "TF-IDF":
            return tf
        # Hybrid formula: Mixes Semantic (Context) and TF-IDF (Keywords) together
        return alpha * sem + (1 - alpha) * tf

    def explain(self, i: int, j: int) -> list[str]:
        # It compares movie A (searched) and movie B (recommended) and finds overlap in cast/director/genres.
        a, b = self.df.iloc[i], self.df.iloc[j]
        reasons = []
        if a["director"] and a["director"] == b["director"]:
            reasons.append("Dir: " + a["director"][0])
        cast = [c for c in a["cast"] if c in b["cast"]]
        if cast:
            reasons.append("Cast: " + ", ".join(cast[:2]))
        genres = [g for g in a["genres"] if g in b["genres"]]
        if genres:
            reasons.append(", ".join(genres[:3]))
        kws = [k for k in a["keywords"] if k in set(b["keywords"])]
        if kws:
            reasons.append("#" + kws[0].replace(" ", ""))
        return reasons

    def recommend(self, i: int, k: int = 10, mode: str = "Hybrid", alpha: float = 0.5,
                  genres: list[str] | None = None, min_rating: float = 0.0) -> list[dict]:
        # 1. Get the raw scores for all ~4800 movies compared to the chosen movie 'i'
        s = self.scores(i, mode, alpha).astype(np.float64)
        mask = np.ones(len(s), dtype=bool)
        # 2. Apply Filters (Genres / Minimum Rating) by turning invalid movies into "False" in our mask
        if genres:
            wanted = set(genres)
            mask &= np.fromiter((bool(g & wanted) for g in self._gsets), bool, len(s))
        if min_rating > 0:
            mask &= self._rating >= min_rating
        # 3. Set score to negative infinity for filtered movies AND for the searched movie itself
        s[~mask] = -np.inf
        s[i] = -np.inf
        # 4. Sort the scores and return the top 'k' results
        top = [j for j in np.argsort(-s)[:k] if np.isfinite(s[j])]
        return [{"idx": int(j), "score": float(s[j]), "reasons": self.explain(i, int(j))}
                for j in top]
