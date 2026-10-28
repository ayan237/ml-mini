"""Content-based music recommender.

Pipeline:
  1. Standardise audio features (StandardScaler)
  2. Cluster songs with KMeans (used for "mood clusters" + diversity)
  3. Rank by cosine similarity to a song, or to the mean vector of several
     liked songs (a simple user taste profile)
  4. Evaluate with genre Precision@K (genre is NOT used as an input feature)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]


class MusicRecommender:
    def __init__(self, df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42):
        self.df = df.reset_index(drop=True).copy()
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df[FEATURES])
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        self.df["cluster"] = self.kmeans.fit_predict(self.X)
        self._idx = {name: i for i, name in enumerate(self.df["track_name"])}

    # ---------- recommendation ----------
    def _rank(self, vector, exclude, k, genre=None, same_cluster=False):
        sims = cosine_similarity(vector.reshape(1, -1), self.X)[0]
        out = self.df.copy()
        out["similarity"] = sims
        out = out.drop(index=list(exclude))
        if genre:
            out = out[out["genre"] == genre]
        if same_cluster:
            cl = self.kmeans.predict(vector.reshape(1, -1))[0]
            out = out[out["cluster"] == cl]
        return out.sort_values("similarity", ascending=False).head(k)

    def recommend_similar(self, track_name, k=10, genre=None, same_cluster=False):
        i = self._idx[track_name]
        return self._rank(self.X[i], [i], k, genre, same_cluster)

    def recommend_for_profile(self, liked_tracks, k=10, genre=None):
        """Taste profile = mean feature vector of the liked songs."""
        ids = [self._idx[t] for t in liked_tracks]
        profile = self.X[ids].mean(axis=0)
        return self._rank(profile, ids, k, genre)

    def recommend_from_sliders(self, prefs: dict, k=10):
        """prefs: raw feature values, e.g. {'energy': 0.8, 'valence': 0.6}.
        Unspecified features default to the dataset mean."""
        raw = self.df[FEATURES].mean().to_dict()
        raw.update(prefs)
        vec = self.scaler.transform(pd.DataFrame([raw])[FEATURES])[0]
        return self._rank(vec, [], k)

    # ---------- evaluation ----------
    def precision_at_k(self, k=10, sample=None, seed=0):
        """Fraction of the top-k recommendations that share the query song's genre."""
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self.df))
        if sample and sample < len(idx):
            idx = rng.choice(idx, sample, replace=False)
        genres = self.df["genre"].values
        scores = []
        for i in idx:
            sims = cosine_similarity(self.X[i].reshape(1, -1), self.X)[0]
            sims[i] = -1
            top = np.argsort(-sims)[:k]
            scores.append((genres[top] == genres[i]).mean())
        return float(np.mean(scores))

    def cluster_purity(self):
        """How genre-pure the KMeans clusters are (1.0 = perfect)."""
        ct = pd.crosstab(self.df["cluster"], self.df["genre"])
        return float(ct.max(axis=1).sum() / ct.values.sum())


def load_data(path="data/songs.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + ["track_name", "artist", "genre"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df.dropna(subset=FEATURES).drop_duplicates("track_name")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    rec = MusicRecommender(load_data())
    print(f"Songs: {len(rec.df)}")
    print(f"Precision@10 (genre match): {rec.precision_at_k(10, sample=300):.3f}")
    print(f"KMeans cluster purity:      {rec.cluster_purity():.3f}")
    seed = rec.df.loc[0, "track_name"]
    print(f"\nTop 5 similar to '{seed}' ({rec.df.loc[0, 'genre']}):")
    print(rec.recommend_similar(seed, 5)[["track_name", "artist", "genre", "similarity"]]
          .to_string(index=False))
