"""
test_retrieval.py — Quick smoke test for MovieRetriever.

Verifies the offline-built index loads correctly and returns
semantically relevant movies for a natural-language query.
Delete after the full pipeline is working.
"""
from rag_pipeline import MovieRetriever


def main():
    retriever = MovieRetriever()

    queries = [
        "a heartwarming family movie about friendship",
        "dark psychological thriller with a twist ending",
        "feel-good romantic comedy for a rainy night",
    ]

    for q in queries:
        print(f"\nQuery: {q!r}")
        print("-" * 70)
        results = retriever.search(q, k=5)
        for m in results:
            print(f"  {m['score']:.3f}  {m['title']:<50}  {m['genres']}")


if __name__ == "__main__":
    main()