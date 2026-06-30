"""
rag_pipeline.py — Online RAG pipeline: retrieve + generate.

"""
import pickle
import numpy as np
import faiss
from openai import AzureOpenAI

import config


# Retriever 

class MovieRetriever:
    """Loads the FAISS index + metadata once, serves repeated queries.
    """

    def __init__(self):
        # Load FAISS index from disk into memory
        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))

        # Load parallel metadata list (position i == FAISS internal ID i)
        with open(config.METADATA_PATH, "rb") as f:
            self.metadata: list[dict] = pickle.load(f)

        # Single Azure client reused for all embedding calls
        self.client = AzureOpenAI(
            azure_endpoint=config.AZURE_ENDPOINT,
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
        )

        # Sanity check: index size must match metadata length
        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                f"Index/metadata mismatch: "
                f"{self.index.ntotal} vectors vs {len(self.metadata)} records. "
                f"Re-run build_index.py to rebuild both."
            )
        print(f"Retriever ready: {self.index.ntotal} movies indexed")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query into a 1536-dim vector."""
        response = self.client.embeddings.create(
            model=config.EMBEDDING_DEPLOYMENT,
            input=query,
        )
        vec = np.array([response.data[0].embedding], dtype="float32")
        faiss.normalize_L2(vec) 
        return vec

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        """Return top-K movies most semantically similar to the query.
        """
        query_vec = self.embed_query(query)
        scores, ids = self.index.search(query_vec, k)

        # scores and ids are 2D arrays of shape (1, k) — take the first row
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1: 
                continue
            movie = dict(self.metadata[idx]) 
            movie["score"] = float(score)
            results.append(movie)
        return results

# Prompt construction

SYSTEM_PROMPT = """You are CineMatch, a movie recommendation assistant.

You'll be given the user's current request, any prior conversation, and a
list of retrieved movies with their genres and plot summaries. Your job:

1. Pick 3 movies from the retrieved list that best match the user's intent.
2. Consider conversation history — if the user is asking for "more like X"
   or rejecting earlier picks, adjust accordingly. NEVER recommend a movie
   you've already recommended in this conversation.
3. For each pick, write a 1-2 sentence reason grounded in the plot summary.
4. NEVER recommend movies not in the retrieved list.
5. NEVER invent plot details — only use what's in the provided summaries.

Format your response as exactly:
**Movie title (year)**: reason

(One blank line between picks. No preamble, no closing remarks.)
"""


def build_user_prompt(query: str, retrieved: list[dict]) -> str:
    """Format the user message with query + retrieved candidates."""
    lines = [f'User request: "{query}"', "", "Retrieved candidates:"]
    for i, m in enumerate(retrieved, 1):
        lines.append(
            f"{i}. {m['title']} | Genres: {m['genres']}\n"
            f"   Plot: {m['intro']}"
        )
    return "\n".join(lines)


# Generation ---------------------------------------------------

class RAGPipeline:
    """End-to-end RAG: retrieve, prompt, generate, with conversation memory.

    Conversation history is held in self.history as OpenAI message dicts.
    Caller can reset() between sessions.
    """

    def __init__(self, retriever: MovieRetriever):
        self.retriever = retriever
        self.client = retriever.client
        self.history: list[dict] = []  # OpenAI message format

    def reset(self) -> None:
        """Clear conversation history (start a fresh session)."""
        self.history = []

    def answer(self, query: str, k: int = config.TOP_K) -> dict:
        """Run the full RAG pipeline for one turn, with history awareness.

        Each call appends both the user query and assistant reply to
        self.history, so the next call has full context.
        """
        retrieved = self.retriever.search(query, k=k)

        # Build user message: include retrieval context inline so the LLM
        # always sees the candidates for THIS turn (history only carries
        # the conversation, not stale retrievals).
        user_msg = build_user_prompt(query, retrieved)

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.history
            + [{"role": "user", "content": user_msg}]
        )

        response = self.client.chat.completions.create(
            model=config.CHAT_DEPLOYMENT,
            messages=messages,
        )
        answer_text = response.choices[0].message.content

        # Append THIS turn to history. We store the raw query (not the full
        # prompt with retrieval) to keep history compact and avoid blowing
        # past context window after many turns.
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer_text})

        return {
            "query": query,
            "answer": answer_text,
            "retrieved": retrieved,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            "turn": len(self.history) // 2,
        }


# CLI entrypoint -----------------------------------------------

def main():
    retriever = MovieRetriever()
    pipeline = RAGPipeline(retriever)

    print("\nCineMatch RAG ready. Type a movie mood/request (or 'quit'):\n")
    while True:
        query = input("> ").strip()
        if query.lower() in {"quit", "exit", ""}:
            break

        result = pipeline.answer(query)
        print("\n" + result["answer"])
        print(
            f"\n[tokens: prompt={result['usage']['prompt_tokens']}, "
            f"completion={result['usage']['completion_tokens']}]\n"
        )


if __name__ == "__main__":
    main()