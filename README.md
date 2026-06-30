# CineMatch-RAG

A movie recommender that uses semantic vector search + a grounded LLM
to answer free-form queries like "I just broke up, want something
cathartic" or "dark psychological thriller with a twist ending."

Built on Azure OpenAI (embeddings + generation) and FAISS for local
vector retrieval. Multi-turn conversation — the LLM keeps context
across turns and won't repeat earlier recommendations.

![CineMatch](system.png)

## Stack

- Python 3.10+
- Azure OpenAI: `text-embedding-3-small` (1536-dim) + `gpt-5-mini`
- FAISS (`IndexFlatIP`, exact cosine search)
- Flask
- MovieLens-1M + IMDB plot summaries (~2,945 movies after filtering)

## How it works

There are two pipelines.

**Offline** (`build_index.py`, run once): load movies, concatenate
title/genres/intro into one string per movie, batch-call the embedding
API, store the resulting 1536-dim vectors in a FAISS index. Metadata
(title, genres, intro) lives in a parallel pickle file — FAISS only
knows vector + integer ID, so we look up the actual movie data by ID.

**Online** (`rag_pipeline.py`, per query): embed the user's query, do a
top-K cosine search in FAISS, pull the matching metadata, drop it all
into a prompt, and let `gpt-5-mini` pick 3 movies and explain why.
The system prompt explicitly forbids inventing plot details or
recommending anything not in the retrieved list.

Conversation history is held in memory as OpenAI message format and
fed back on every turn, so follow-ups like "give me three more, but
lighter" actually work.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your Azure endpoint, key, and deployment names
python build_index.py     # one-time, ~2 minutes
python app.py
# open http://127.0.0.1:5000/
```

You'll need MovieLens-1M under `data/ml-1m/` —
[download here](https://grouplens.org/datasets/movielens/1m/).

## Evaluation

`evaluate.py` runs 10 synthetic queries with genre-based heuristic
ground truth (a result counts as relevant if its genres overlap with
the query's intended genre).

| Metric | Value |
|---|---|
| recall@1 | 0.900 |
| recall@5 | 0.900 |
| recall@10 | 0.890 |
| Retrieval latency mean / P95 | 375 ms / 1,132 ms |
| End-to-end latency mean / P95 | 8.1 s / 11.4 s |
| Cost per query | ~$0.0018 |


## Local MVP vs production

| Layer | Now (local) | Production target |
|---|---|---|
| Embedding + LLM | Azure OpenAI | Azure OpenAI |
| Vector store | FAISS | Azure AI Search (vector + BM25 + filters) |
| Structured data | pickle | Cosmos DB |
| Offline indexing | local script | Azure Databricks |
| App hosting | Flask localhost | Azure App Service |

The `MovieRetriever` class is an interface — swapping FAISS for Azure
AI Search means replacing one implementation, not redesigning the
pipeline. That was a deliberate choice during the MVP; at ~3,000
movies a cloud vector store would have been overkill (FAISS searches
in under 10 ms), but designing for the migration kept the option open.

## A few engineering notes

**Why `IndexFlatIP` not `IndexFlatL2`?** OpenAI embeddings are
L2-normalized, so inner product equals cosine similarity. L2 distance
would be the wrong metric on normalized vectors.

**Why batch embeddings 64 at a time?** The API call's latency is
mostly network. Batching 3,000 inputs into ~47 calls vs 3,000 calls
turns minutes of waiting into under a minute.

**Why no SVD / collaborative filtering?** The legacy version of this
project had SVD. I dropped it for the MVP because the UI takes free
text with no user ID — SVD's value is using rating history, and
without login there's nothing to use. Adding SVD as a candidate
generation layer for logged-in users is the first item in future work.

**A bug worth mentioning.** The LLM was supposed to pick 3 movies but
the UI sometimes only showed 2. Turned out MovieLens stores titles in
library style (`Dark Half, The`) and the LLM helpfully "corrected"
them to natural grammar (`The Dark Half`) in its output, so my title
matching missed them. Fixed with a normalization function (lowercase,
strip year, un-invert articles). The deeper fix would be to use JSON
mode and have the LLM return movie IDs directly — strings are a bad
join key when one side is an LLM.

## Future work

In rough priority order:

1. Add SVD as a candidate generation layer for logged-in users.
2. Switch the LLM to JSON mode + return movie IDs (kills the title
   matching problem).
3. Run RAGAS faithfulness for automated hallucination detection.
4. Migrate vector store to Azure AI Search at scale.
5. Switch generation model to a non-reasoning one for ~10× cost cut.
6. Persist conversations in Cosmos DB.

## Data

- [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) from
  GroupLens Research.
- Plot summaries scraped from IMDB.
