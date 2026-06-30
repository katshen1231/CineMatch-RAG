"""
build_index.py — Offline pipeline: movies → embeddings → FAISS index.

Run once before serving. Re-run whenever movie data changes.

Pipeline:
    1. Load MovieLens-1M titles/genres + IMDB intro CSV
    2. Build a single searchable text string per movie
    3. Batch-call Azure OpenAI embeddings (text-embedding-3-small, 1536-dim)
    4. Insert vectors into a FAISS index
    5. Persist index + metadata to disk

Output:
    indexes/movies.faiss   FAISS index (vectors only)
    indexes/metadata.pkl   Parallel array of movie metadata dicts
"""
import pickle
import time
import numpy as np
import pandas as pd
import faiss
from openai import AzureOpenAI
from tqdm import tqdm

import config


# Load and merge raw data 

def load_movies() -> pd.DataFrame:
    movies = pd.read_csv(
        config.MOVIES_PATH,
        sep="::",
        engine="python",
        header=None,
        names=["movieId", "title", "genres"],
        encoding="ISO-8859-1",
    )

    intros = pd.read_csv(config.INTRO_PATH, encoding="utf-8")
    intros = intros.rename(columns={"id": "movieId"})
    intros["movieId"] = intros["movieId"].astype(int)

    # Left-join intros onto movies (some movies have no intro)
    df = movies.merge(intros[["movieId", "intro"]], on="movieId", how="left")

    # Drop movies without an intro — they have no semantic content to embed
    before = len(df)
    df = df.dropna(subset=["intro"]).reset_index(drop=True)
    print(f"Loaded {len(df)} movies with intros (dropped {before - len(df)} without)")

    return df


def build_search_text(row: pd.Series) -> str:
    title = row["title"]
    genres = row["genres"].replace("|", ", ")
    intro = row["intro"]
    return f"{title}. Genres: {genres}. {intro}"


# Batch-embed movies via Azure OpenAI 
def get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )


def embed_batch(client: AzureOpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=config.EMBEDDING_DEPLOYMENT,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def embed_all_movies(df: pd.DataFrame) -> np.ndarray:
    """Embed every movie in df, returning an (N, EMBEDDING_DIM) matrix.
    """
    client = get_azure_client()
    texts = [build_search_text(row) for _, row in df.iterrows()]

    all_vectors: list[np.ndarray] = []
    batch_size = config.EMBED_BATCH_SIZE

    # tqdm wraps a range to display a progress bar — pure UX, no logic
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding movies"):
        batch = texts[i : i + batch_size]

        # Retry up to 3 times on transient failures
        for attempt in range(3):
            try:
                vectors = embed_batch(client, batch)
                all_vectors.append(vectors)
                break
            except Exception as e:
                if attempt == 2:
                    # Final attempt failed — abort with context
                    raise RuntimeError(
                        f"Embedding failed at batch {i}-{i+batch_size}: {e}"
                    ) from e
                wait = 2 ** attempt  
                print(f"  retry {attempt + 1}/3 after {wait}s: {e}")
                time.sleep(wait)

    # Vertically stack all batch matrices into a single (N, dim) matrix
    matrix = np.vstack(all_vectors)
    print(f"Embedded {matrix.shape[0]} movies, vector dim = {matrix.shape[1]}")
    return matrix

# Build FAISS index and persist

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Wrap vectors in a FAISS IndexFlatIP for exact cosine similarity search.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Defensive normalization — OpenAI outputs are already unit vectors,
    # but doing this explicitly makes the math contract obvious in code.
    faiss.normalize_L2(vectors)

    index.add(vectors)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}, "
          f"type=IndexFlatIP")
    return index


def build_metadata(df: pd.DataFrame) -> list[dict]:
    """Build the metadata list parallel to FAISS internal IDs.

    Position i in this list corresponds to FAISS internal ID i.
    Only fields needed downstream (display, filtering, grounding) are kept.
    """
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "intro": row["intro"],
        })
    return metadata


def save_artifacts(index: faiss.Index, metadata: list[dict]) -> None:
    """Persist FAISS index and metadata to disk."""
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved index    -> {config.FAISS_INDEX_PATH}")
    print(f"Saved metadata -> {config.METADATA_PATH}")


# --- Main pipeline ---------------------------------------------------------

def main() -> None:
    start = time.time()

    df = load_movies()
    vectors = embed_all_movies(df)
    index = build_faiss_index(vectors)
    metadata = build_metadata(df)
    save_artifacts(index, metadata)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. "
          f"Index ready for retrieval at {config.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
