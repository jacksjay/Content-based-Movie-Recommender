import html
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests
import streamlit as st

from src.recommender import MODES, Recommender

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg,#7c5cff,#ff5c93); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}
.hero p {color:#8b93a7; margin:.2rem 0 1.2rem 0;}
.card {background:#161b26; border:1px solid #232a3a; border-radius:14px; overflow:hidden;
  transition: transform .15s ease, border-color .15s ease; margin-bottom:.5rem;}
.card:hover {transform: translateY(-3px); border-color:#7c5cff;}
.poster {width:100%; aspect-ratio:2/3; object-fit:cover; display:block;}
.ph {aspect-ratio:2/3; display:flex; align-items:center; justify-content:center;
  font-size:2.4rem; font-weight:800; color:rgba(255,255,255,.85);}
.cbody {padding:.7rem .8rem .8rem .8rem;}
.ctitle {font-weight:600; font-size:.92rem; line-height:1.25; min-height:2.3em;}
.meta {color:#8b93a7; font-size:.76rem; margin-top:.15rem;}
.match {float:right; background:rgba(124,92,255,.18); color:#b6a4ff; border-radius:999px;
  padding:.05rem .5rem; font-size:.72rem; font-weight:600;}
.chip {display:inline-block; background:#232a3a; color:#c9cfe0; border-radius:999px;
  padding:.1rem .5rem; font-size:.68rem; margin:.25rem .2rem 0 0;}
.big-chip {display:inline-block; background:#232a3a; color:#c9cfe0; border-radius:999px;
  padding:.2rem .7rem; font-size:.8rem; margin:.2rem .3rem 0 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading notebook results...")
def load() -> Recommender:
    return Recommender.load()


def tmdb_key() -> str | None:
    key = os.getenv("TMDB_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("TMDB_API_KEY")
    except Exception:
        return None


@lru_cache(maxsize=4096)
def fetch_poster(movie_id: int, api_key: str) -> str | None:
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                         params={"api_key": api_key}, timeout=5)
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:
        return None


def posters_for(ids: list[int]) -> dict[int, str | None]:
    key = tmdb_key()
    if not key:
        return {i: None for i in ids}
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(ids, ex.map(lambda i: fetch_poster(i, key), ids)))


def visual(row, poster: str | None) -> str:
    if poster:
        return f'<img class="poster" src="{poster}">'
    hue = (int(row["movie_id"]) * 47) % 360
    initials = "".join(w[0] for w in row["title"].split()[:2]).upper()
    return (f'<div class="ph" style="background:linear-gradient(135deg,'
            f'hsl({hue},55%,38%),hsl({(hue + 50) % 360},60%,18%))">{html.escape(initials)}</div>')


def meta_line(row) -> str:
    parts = []
    if row["year"] == row["year"]:
        parts.append(str(int(row["year"])))
    if row["runtime"] == row["runtime"] and row["runtime"] > 0:
        parts.append(f'{int(row["runtime"])} min')
    parts.append(f'⭐ {row["vote_average"]:.1f}')
    return " · ".join(parts)


def card(row, poster, score, reasons) -> str:
    chips = "".join(f'<span class="chip">{html.escape(r)}</span>' for r in reasons[:3])
    return (
        '<div class="card">' + visual(row, poster) + '<div class="cbody">'
        f'<span class="match">{score * 100:.0f}%</span>'
        f'<div class="ctitle">{html.escape(row["title"])}</div>'
        f'<div class="meta">{meta_line(row)}</div>{chips}</div></div>'
    )


def select_movie(label: str) -> None:
    st.session_state["selected_label"] = label


# ---------- app ----------
try:
    rec = load()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.info("Run the export cell at the end of your notebook, unzip `artifacts.zip` into this "
            "project's `artifacts/` folder, then restart the app.")
    st.stop()

df = rec.df

st.markdown('<div class="hero"><h1>🎬 CineMatch</h1>'
            '<p>Movie recommendations powered by TF-IDF + semantic sentence embeddings</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Ranking model", MODES, help="Hybrid blends semantic and TF-IDF scores")
    alpha = 0.5
    if mode == "Hybrid":
        alpha = st.slider("Semantic weight (α)", 0.0, 1.0, 0.5, 0.05,
                          help="0 = pure TF-IDF, 1 = pure semantic")
    k = st.slider("Number of results", 5, 20, 10, 5)
    genres = st.multiselect("Only show genres", rec.all_genres)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, 0.5)
    st.caption(f"{len(df):,} movies in catalog")
    if not tmdb_key():
        st.caption("💡 Set `TMDB_API_KEY` to show real posters.")

if "selected_label" not in st.session_state:
    default = next((l for l in rec.labels if l.startswith("Avatar")), rec.labels[0])
    st.session_state["selected_label"] = default

label = st.selectbox("🔍 Search a movie you like (type to filter)", rec.labels,
                     key="selected_label")
i = rec.label_to_idx[label]
row = df.iloc[i]

results = rec.recommend(i, k=k, mode=mode, alpha=alpha, genres=genres, min_rating=min_rating)
ids = [int(row["movie_id"])] + [int(df.iloc[r["idx"]]["movie_id"]) for r in results]
posters = posters_for(ids)

tab_rec, tab_cmp = st.tabs(["✨ Recommendations", "🧪 Compare models"])

with tab_rec:
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown('<div class="card">' + visual(row, posters[ids[0]]) + '</div>',
                    unsafe_allow_html=True)
    with right:
        st.subheader(row["title"])
        st.caption(meta_line(row))
        chips = "".join(f'<span class="big-chip">{html.escape(g)}</span>' for g in row["genres"])
        st.markdown(chips, unsafe_allow_html=True)
        st.write(row["overview"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Director", row["director"][0] if row["director"] else "—")
        c2.metric("Lead cast", row["cast"][0] if row["cast"] else "—")
        c3.metric("Votes", f'{int(row["vote_count"]):,}')

    st.markdown(f"### Because you picked *{row['title']}*")
    if not results:
        st.warning("No movies match these filters. Try relaxing genre or rating.")
    for start in range(0, len(results), 5):
        cols = st.columns(5, gap="medium")
        for col, r in zip(cols, results[start:start + 5]):
            rrow = df.iloc[r["idx"]]
            with col:
                st.markdown(card(rrow, posters[int(rrow["movie_id"])], r["score"], r["reasons"]),
                            unsafe_allow_html=True)
                st.button("Explore →", key=f"go_{rrow['movie_id']}_{start}",
                          on_click=select_movie, args=(rrow["label"],),
                          use_container_width=True)

with tab_cmp:
    st.write("Top 5 from each model for the selected movie. Disagreements show where "
             "semantic understanding differs from keyword overlap.")
    cols = st.columns(3, gap="large")
    for col, m in zip(cols, MODES):
        with col:
            st.markdown(f"**{m}**")
            for n, r in enumerate(rec.recommend(i, k=5, mode=m, alpha=alpha), 1):
                st.markdown(f"{n}. {df.iloc[r['idx']]['title']}  `{r['score']:.2f}`")


