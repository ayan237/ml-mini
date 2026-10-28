import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from recommender import FEATURES, MusicRecommender, load_data

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")


@st.cache_resource
def get_recommender():
    if not os.path.exists("data/songs.csv"):
        os.system("python generate_data.py")
    return MusicRecommender(load_data())


rec = get_recommender()
df = rec.df

st.title("🎵 Music Recommendation System")
st.caption("Content-based filtering: StandardScaler + cosine similarity + KMeans clustering")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Similar songs", "My taste profile", "Mood sliders", "Model insights"])

genres = ["All"] + sorted(df["genre"].unique())
show_cols = ["track_name", "artist", "genre", "similarity"]


def show(results):
    res = results.copy()
    res["similarity"] = res["similarity"].round(3)
    st.dataframe(res[show_cols].reset_index(drop=True), use_container_width=True)


# ---- Tab 1
with tab1:
    c1, c2, c3 = st.columns([3, 1, 1])
    song = c1.selectbox("Pick a song you like", df["track_name"].sort_values())
    k = c2.slider("How many?", 5, 20, 10, key="k1")
    g = c3.selectbox("Filter genre", genres, key="g1")
    same_cluster = st.checkbox("Only songs from the same mood cluster")
    if st.button("Recommend", key="b1"):
        row = df[df["track_name"] == song].iloc[0]
        st.write(f"**{song}** — {row['artist']} · {row['genre']}")
        show(rec.recommend_similar(song, k, None if g == "All" else g, same_cluster))

# ---- Tab 2
with tab2:
    liked = st.multiselect("Select a few songs you like (2 or more works best)",
                           df["track_name"].sort_values())
    k2 = st.slider("How many?", 5, 20, 10, key="k2")
    g2 = st.selectbox("Filter genre", genres, key="g2")
    if st.button("Build my profile & recommend", key="b2"):
        if not liked:
            st.warning("Pick at least one song.")
        else:
            show(rec.recommend_for_profile(liked, k2, None if g2 == "All" else g2))

# ---- Tab 3
with tab3:
    st.write("Describe the vibe you want:")
    c1, c2, c3 = st.columns(3)
    prefs = {
        "energy": c1.slider("Energy", 0.0, 1.0, 0.6),
        "danceability": c2.slider("Danceability", 0.0, 1.0, 0.6),
        "valence": c3.slider("Positivity (valence)", 0.0, 1.0, 0.5),
        "acousticness": c1.slider("Acousticness", 0.0, 1.0, 0.3),
        "instrumentalness": c2.slider("Instrumentalness", 0.0, 1.0, 0.1),
        "tempo": c3.slider("Tempo (BPM)", 60, 200, 110),
    }
    if st.button("Find songs", key="b3"):
        show(rec.recommend_from_sliders(prefs, 10))

# ---- Tab 4
with tab4:
    c1, c2 = st.columns(2)
    c1.metric("Precision@10 (genre match)", f"{rec.precision_at_k(10, sample=300):.2%}")
    c2.metric("KMeans cluster purity", f"{rec.cluster_purity():.2%}")
    st.caption("Genres overlap in audio-feature space (e.g. energetic Pop sounds like Rock), "
               "so these scores are modest on real music. Genre is only a proxy label.")
    pca = PCA(n_components=2).fit_transform(rec.X)
    plot_df = df.assign(PC1=pca[:, 0], PC2=pca[:, 1], cluster=df["cluster"].astype(str))
    color = st.radio("Colour by", ["genre", "cluster"], horizontal=True)
    fig = px.scatter(plot_df, x="PC1", y="PC2", color=color,
                     hover_data=["track_name", "artist", "genre"], opacity=0.7,
                     title="Songs in audio-feature space (PCA, 2 components)")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Average audio features per genre")
    st.dataframe(df.groupby("genre")[FEATURES].mean().round(2), use_container_width=True)
