"""
evaluate.py — Quantitative evaluation of the RAG pipeline.

Metrics:
    1. precision@k — heuristic: among the top-K retrieved movies, what
       fraction have genres overlapping with the query's intended genre.
       We use precision rather than recall because for genre-based
       queries, the relevant set in MovieLens-1M can contain hundreds
       of movies, making recall@10 uninformatively small. Precision@k
       better captures "ranking quality in what the user actually sees".

    2. latency  — wall time for the retrieval step and the full
       end-to-end pipeline (retrieval + LLM generation).

    3. cost     — average prompt + completion tokens per query, with
       a rough USD estimate based on text-embedding-3-small and
       gpt-5-mini list prices.

This evaluation uses SYNTHETIC queries paired with a target genre.
Full evaluation would require human-annotated relevance judgments or
LLM-as-judge metrics (e.g. RAGAS faithfulness / answer-relevance) —
this approximation is fast and captures coarse semantic alignment.
"""
import time
import statistics
from rag_pipeline import MovieRetriever, RAGPipeline


# Pricing as of June 2026 (per 1M tokens). Update if rates change.
EMBED_PRICE_PER_1M = 0.02   # text-embedding-3-small
LLM_INPUT_PER_1M   = 0.25   # gpt-5-mini estimate
LLM_OUTPUT_PER_1M  = 2.00   # gpt-5-mini estimate (incl. reasoning tokens)


# Synthetic query set: natural-language queries paired with the genre
# we expect a relevant movie to contain. Designed to span moods,
# tones, and intents — not just literal genre keywords.
EVAL_SET = [
    ("a heartwarming family movie about friendship",       "Children's"),
    ("dark psychological thriller with a twist ending",    "Thriller"),
    ("feel-good romantic comedy for a rainy night",        "Romance"),
    ("epic science-fiction adventure with stunning visuals", "Sci-Fi"),
    ("classic horror film that built the genre",           "Horror"),
    ("animated movie kids will love",                      "Animation"),
    ("noir detective story with morally grey characters",  "Film-Noir"),
    ("documentary about something fascinating",            "Documentary"),
    ("western with gunfights and a lonely hero",           "Western"),
    ("musical with memorable songs",                       "Musical"),
]

K_VALUES = [1, 5, 10]


def precision_at_k(retrieved: list[dict], target_genre: str, k: int) -> float:
    """Fraction of top-K retrieved results whose genres contain the target genre.

    This is precision@k: among the K results we surface, how many are
    on-topic. We use genre overlap as a heuristic proxy for topical
    relevance — a result counts as a "hit" if its genres list contains
    the query's intended genre.

    Note on metric choice: we report precision rather than recall because
    true recall@k = (hits in top K) / (total relevant in dataset). For
    genre-based queries on MovieLens-1M the denominator can be in the
    hundreds, making recall@10 small and uninformative. Precision@k
    captures what users actually experience: ranking quality in the
    top results.

    Returns a value in [0, 1] — higher is better.
    """
    top_k = retrieved[:k]
    hits = sum(1 for m in top_k if target_genre in m["genres"])
    return hits / k


def evaluate_retrieval_only(retriever: MovieRetriever) -> dict:
    """Measure retrieval precision@k and latency (no LLM call)."""
    precisions = {k: [] for k in K_VALUES}
    latencies = []

    for query, target_genre in EVAL_SET:
        t0 = time.perf_counter()
        retrieved = retriever.search(query, k=max(K_VALUES))
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

        for k in K_VALUES:
            precisions[k].append(precision_at_k(retrieved, target_genre, k))

    return {
        "precision_at_k": {k: statistics.mean(precisions[k]) for k in K_VALUES},
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p95":  sorted(latencies)[int(len(latencies) * 0.95)],
        "n_queries": len(EVAL_SET),
    }


def evaluate_end_to_end(pipeline: RAGPipeline) -> dict:
    """Measure full RAG latency + token costs.

    Resets pipeline history between queries so each call is isolated.
    """
    latencies = []
    prompt_tokens = []
    completion_tokens = []

    for query, _ in EVAL_SET:
        pipeline.reset()
        t0 = time.perf_counter()
        result = pipeline.answer(query)
        latencies.append((time.perf_counter() - t0) * 1000)
        prompt_tokens.append(result["usage"]["prompt_tokens"])
        completion_tokens.append(result["usage"]["completion_tokens"])

    avg_prompt = statistics.mean(prompt_tokens)
    avg_completion = statistics.mean(completion_tokens)

    cost_per_query_usd = (
        avg_prompt * LLM_INPUT_PER_1M / 1_000_000
        + avg_completion * LLM_OUTPUT_PER_1M / 1_000_000
    )

    return {
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p95":  sorted(latencies)[int(len(latencies) * 0.95)],
        "tokens_prompt_mean":     round(avg_prompt),
        "tokens_completion_mean": round(avg_completion),
        "cost_per_query_usd":     round(cost_per_query_usd, 5),
    }


def main():
    print("=" * 60)
    print(f"CineMatch-RAG evaluation ({len(EVAL_SET)} queries)")
    print("=" * 60)

    retriever = MovieRetriever()
    pipeline = RAGPipeline(retriever)

    print("\n[1/2] Retrieval-only metrics...")
    retr = evaluate_retrieval_only(retriever)
    for k, v in retr["precision_at_k"].items():
        print(f"      precision@{k:<2} = {v:.3f}")
    print(f"      latency mean = {retr['latency_ms_mean']:.1f} ms")
    print(f"      latency p95  = {retr['latency_ms_p95']:.1f} ms")

    print("\n[2/2] End-to-end metrics (retrieval + LLM)...")
    e2e = evaluate_end_to_end(pipeline)
    print(f"      latency mean           = {e2e['latency_ms_mean']:.1f} ms")
    print(f"      latency p95            = {e2e['latency_ms_p95']:.1f} ms")
    print(f"      prompt tokens mean     = {e2e['tokens_prompt_mean']}")
    print(f"      completion tokens mean = {e2e['tokens_completion_mean']}")
    print(f"      cost per query         = ${e2e['cost_per_query_usd']:.5f}")

    print()
    print("Summary:")
    print("-" * 60)
    print(f"  precision@10 = {retr['precision_at_k'][10]:.3f}  "
          f"(heuristic: genre overlap on {len(EVAL_SET)} synthetic queries)")
    print(f"  retrieval latency P95 = {retr['latency_ms_p95']:.0f} ms")
    print(f"  end-to-end latency mean = {e2e['latency_ms_mean']:.0f} ms")
    print(f"  avg cost per query = ${e2e['cost_per_query_usd']:.5f}")


if __name__ == "__main__":
    main()