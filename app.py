"""
app.py — Flask backend for CineMatch-RAG.

Single-page web UI wired to the offline-built FAISS index + Azure OpenAI.
The RAG pipeline is loaded ONCE at startup and reused across requests
(loading FAISS + metadata + Azure client costs ~1s per instance).

Routes:
    GET  /                 -> single-page UI
    GET  /random_movies    -> random posters for the left-side scroller
    POST /recommend        -> RAG-grounded recommendation (LLM + retrieval)
    GET  /poster/<file>    -> poster image, default fallback if missing
"""
import random
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

import config
from rag_pipeline import MovieRetriever, RAGPipeline


# --- Initialize the pipeline once at startup ------------------------------
# Doing this at module top-level (not inside a route) ensures the heavy
# load happens during boot, not on the first incoming request.
print("Loading RAG pipeline...")
retriever = MovieRetriever()
pipeline = RAGPipeline(retriever)
print("Pipeline ready.\n")

POSTER_DIR = config.DATA_DIR / "IMDBPoster" / "poster"
DEFAULT_POSTER = "default_poster.jpg"

# Serve static files (index.html, background images) from project root
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    """Serve the single-page UI."""
    return send_file("index.html")


@app.route("/random_movies")
def random_movies():
    """Return N random movies for the left-side poster scroller.

    Sampled from the indexed movie set (those with intros), so every
    poster the user sees corresponds to a movie that can be recommended.
    """
    n = min(int(request.args.get("num_movies", 50)), 50)
    sample = random.sample(retriever.metadata, k=min(n, len(retriever.metadata)))

    payload = [
        {
            "movieId": m["movieId"],
            "title": m["title"],
            "poster_path": f"http://127.0.0.1:5000/poster/{m['movieId']}.jpg",
        }
        for m in sample
    ]
    return jsonify({"movies": payload})


@app.route("/recommend", methods=["POST"])
def recommend():
    """Run the RAG pipeline on a free-form user query.

    Request body: { "query": "I just broke up, want something cathartic" }
    Response:     {
        "answer":   "...LLM-generated grounded recommendation...",
        "retrieved": [ {movieId, title, genres, intro, score, poster_path}, ... ],
        "usage":    { prompt_tokens, completion_tokens }
    }
    """
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    result = pipeline.answer(query)

    # Attach poster URLs to retrieved movies so the frontend can display them
    for m in result["retrieved"]:
        m["poster_path"] = f"http://127.0.0.1:5000/poster/{m['movieId']}.jpg"

    return jsonify(result)

@app.route("/reset", methods=["POST"])
def reset():
    """Clear conversation history (start a new session)."""
    pipeline.reset()
    return jsonify({"status": "reset"})

@app.route("/poster/<filename>")
def serve_poster(filename):
    """Serve a poster, falling back to default_poster.jpg if missing.

    The fallback is essential because ~945 of 3883 MovieLens movies have
    no scraped poster — without it, ~25% of cards would show broken images.
    """
    target = POSTER_DIR / filename
    if not target.exists():
        target = POSTER_DIR / DEFAULT_POSTER
    return send_file(target)


if __name__ == "__main__":
    print("CineMatch-RAG running at http://127.0.0.1:5000/")
    # debug=False because debug mode reloads on file change, which would
    # re-load the 18 MB FAISS index every time we edit Python — annoying.
    app.run(debug=False, host="127.0.0.1", port=5000)