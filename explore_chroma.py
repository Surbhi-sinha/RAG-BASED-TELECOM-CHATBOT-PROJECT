"""
Explore the Chroma vector store in a browser GUI.

Two tabs:
  1. Browse  - documents + metadata for each collection, as tables
  2. 2D map  - embeddings reduced to 2D (PCA or t-SNE) and plotted,
               colored by source/category so you can see how they cluster

Run:
    .venv/bin/python -m streamlit run explore_chroma.py
    # or, with the venv activated:  streamlit run explore_chroma.py

Uses only libraries already installed: streamlit, chromadb, scikit-learn,
pandas, numpy. No extra pip installs needed.
"""
import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

CHROMA_DIR = "chroma_store"

st.set_page_config(page_title="Chroma Explorer", page_icon="🔎", layout="wide")


@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


@st.cache_data
def load_collection(name: str):
    """Return docs, metadatas, and embeddings for a collection as a DataFrame + array."""
    col = get_client().get_collection(name)
    data = col.get(include=["documents", "metadatas", "embeddings"])
    df = pd.DataFrame({
        "id": data["ids"],
        "document": data["documents"],
    })
    # flatten metadata dicts into columns
    metas = data["metadatas"] or [{} for _ in data["ids"]]
    meta_df = pd.json_normalize(metas)
    df = pd.concat([df, meta_df], axis=1)
    embeddings = np.array(data["embeddings"]) if data["embeddings"] is not None else None
    return df, embeddings


client = get_client()
collections = [c.name for c in client.list_collections()]

st.title("🔎 Chroma Vector Store Explorer")

if not collections:
    st.warning(
        f"No collections found in '{CHROMA_DIR}/'. "
        "Run the ingestion scripts first (ingest_faq.py, ingest_tickets.py, ingest_pdf.py)."
    )
    st.stop()

st.caption(f"Store: `{CHROMA_DIR}` · Collections: {', '.join(collections)}")

browse_tab, map_tab = st.tabs(["📄 Browse", "🗺️ 2D map"])

# ── Browse tab ────────────────────────────────────────────────────────────────
with browse_tab:
    chosen = st.selectbox("Collection", collections, key="browse_col")
    df, _ = load_collection(chosen)
    st.write(f"**{len(df)} documents** in `{chosen}`")
    st.dataframe(df, use_container_width=True, height=500)

# ── 2D map tab ────────────────────────────────────────────────────────────────
with map_tab:
    st.write(
        "Embeddings are 384-dimensional, so they're reduced to 2D here. "
        "Points that sit close together have similar meaning — that's what the "
        "retriever uses to find relevant context."
    )

    picked = st.multiselect(
        "Collections to plot", collections, default=collections, key="map_cols"
    )
    method = st.radio(
        "Reduction method",
        ["PCA (instant)", "t-SNE (slower, better clusters)"],
        horizontal=True,
    )

    if not picked:
        st.info("Pick at least one collection.")
        st.stop()

    frames, vectors = [], []
    for name in picked:
        df, emb = load_collection(name)
        if emb is None or len(emb) == 0:
            continue
        df = df.copy()
        df["collection"] = name
        # short preview for hover/label
        df["preview"] = df["document"].str.slice(0, 120) + "…"
        frames.append(df)
        vectors.append(emb)

    if not frames:
        st.warning("No embeddings to plot.")
        st.stop()

    all_df = pd.concat(frames, ignore_index=True)
    X = np.vstack(vectors)

    if method.startswith("PCA"):
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        # perplexity must be < n_samples
        perplexity = min(30, max(2, len(X) - 1))
        coords = TSNE(
            n_components=2, random_state=42, perplexity=perplexity, init="pca"
        ).fit_transform(X)

    all_df["x"] = coords[:, 0]
    all_df["y"] = coords[:, 1]

    st.scatter_chart(
        all_df,
        x="x",
        y="y",
        color="collection",
        height=600,
    )

    with st.expander("See the plotted points and their text"):
        st.dataframe(
            all_df[["collection", "preview", "x", "y"]],
            use_container_width=True,
            height=400,
        )
