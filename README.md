# Music Recommendation System (ML Mini Project)

A content-based music recommender with a Streamlit web UI.

## How it works
1. **Features**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo.
2. **Preprocessing**: `StandardScaler` so tempo/loudness don't dominate.
3. **Clustering**: `KMeans` groups songs into "mood clusters".
4. **Recommendation**: cosine similarity between a song (or the mean vector of several liked songs = user taste profile, or slider values) and every other song.
5. **Evaluation**: Precision@K using genre as a proxy label (genre is *not* fed to the model), plus cluster purity.

## Run
```bash
pip install -r requirements.txt
python generate_data.py        # creates data/songs.csv (120 real songs)
python recommender.py          # CLI demo + evaluation metrics
streamlit run app.py           # web app
```

## Dataset
`generate_data.py` builds `data/songs.csv` from 120 real, well-known songs (12 each in Pop, Rock, Hip-Hop, Classical, Jazz, Electronic, Folk, Bollywood, Metal, R&B).
**The audio-feature values are approximate estimates, not exact Spotify numbers.** That is fine for a demo, but for your final submission use real data:

Download the Kaggle "Spotify Tracks Dataset" and save it as `data/songs.csv`.
It needs columns `track_name, artist, genre` (rename `artists`/`track_genre` if needed) and the 9 audio features above. Nothing else changes.

## A note on evaluation
Genre Precision@K is a weak metric for real music: genres overlap heavily in audio-feature space (a high-energy Pop song really does sound like Rock or Metal). On the 120-song set it is about 29% at K=10, and that is expected rather than a bug. Discuss this in your report, and consider extra metrics such as intra-list diversity or a small user study.

## Project structure
```
generate_data.py          builds data/songs.csv from the curated real-song list
recommender.py            ML core (scaling, KMeans, cosine similarity, evaluation)
app.py                    Streamlit UI (4 tabs)
PROJECT_DOCUMENTATION.md  Comprehensive technical and mathematical documentation
```

## Ideas to extend (for extra marks)
- Hybrid model: add collaborative filtering (user-item matrix + SVD)
- Compare cosine vs. Euclidean vs. KNN (`NearestNeighbors`)
- Elbow/silhouette analysis to choose the number of clusters
- Add Spotify API for album art and 30-second previews
