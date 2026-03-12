"""
InteliCore — API Server
========================
Flask API that connects the frontend chat interface
to the RAG-powered persona agent backend.

Endpoints:
  POST /api/chat     — Send a message and get a persona response
  GET  /api/status   — Check server and model health
  POST /api/reset    — Reset conversation history
  GET  /api/sources  — Get sources from last retrieval

Run:
  python server.py
  Then open index.html in your browser (or use Live Server)
"""

import os
import sys
import time
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context
from flask_cors import CORS
import json

import ollama
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from config import Config

# ──────────────────────────────────────────────
#  Initialize
# ──────────────────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # Allow frontend to call API from any origin

cfg = Config()

print("=" * 50)
print("  🧠 InteliCore API Server")
print(f"  Persona: {cfg.persona_name}")
print(f"  Model:   {cfg.llm_model}")
print("=" * 50)

# --- Load embedding model ---
print(f"\n📦 Loading embedding model: {cfg.embedding_model}...")
embedder = SentenceTransformer(cfg.embedding_model)

# --- Load ChromaDB ---
print(f"📦 Loading vector database...")
chroma_client = PersistentClient(path=os.path.abspath(cfg.chroma_db_path))
collection = chroma_client.get_or_create_collection(cfg.collection_name)
doc_count = collection.count()
print(f"📚 Loaded {doc_count} chunks from ChromaDB.")

if doc_count == 0:
    print("⚠️  WARNING: Database is empty! Run `python build_chroma.py` first.")

# --- Conversation history (per-session, in-memory) ---
conversation_history = []
last_sources = []


# ──────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────

def retrieve_context(query: str):
    """Retrieve relevant transcript chunks for a query."""
    global last_sources

    if collection.count() == 0:
        last_sources = []
        return ""

    q_embed = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embed,
        n_results=min(cfg.n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    sources = set()

    if results and results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 / (1 + dist)
            if similarity >= cfg.min_relevance_score:
                chunks.append(doc)
                if meta and "source" in meta:
                    sources.add(meta["source"])

    last_sources = sorted(sources)
    return "\n\n---\n\n".join(chunks) if chunks else ""


def format_history():
    """Format recent conversation history for the prompt."""
    if not conversation_history:
        return ""
    lines = []
    for h in conversation_history[-5:]:
        lines.append(f"User: {h['user']}")
        lines.append(f"You: {h['assistant']}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  API Routes
# ──────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Send a message and get a persona response.

    Request JSON:
      { "message": "What is your leadership style?" }

    Response JSON:
      { "response": "...", "sources": [...], "credits_used": 1 }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    query = data["message"].strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400

    try:
        # Retrieve context from ChromaDB
        context = retrieve_context(query)

        # Build prompt
        history_str = format_history()
        system_prompt = cfg.build_prompt(context, query, history_str)

        # Call Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = ollama.chat(
            model=cfg.llm_model,
            messages=messages,
            options={"temperature": cfg.temperature},
        )

        reply = response["message"]["content"]

        # Save to conversation history
        conversation_history.append({"user": query, "assistant": reply})
        if len(conversation_history) > cfg.max_conversation_history:
            conversation_history.pop(0)

        return jsonify({
            "response": reply,
            "sources": last_sources,
            "credits_used": 1,
        })

    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    Send a message and get a streamed persona response (Server-Sent Events).

    Request JSON:
      { "message": "What is your leadership style?" }

    Response: text/event-stream with JSON chunks
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    query = data["message"].strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400

    # Retrieve context
    context = retrieve_context(query)
    history_str = format_history()
    system_prompt = cfg.build_prompt(context, query, history_str)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    def generate():
        full_response = []
        try:
            stream = ollama.chat(
                model=cfg.llm_model,
                messages=messages,
                options={"temperature": cfg.temperature},
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Final message with complete response and sources
            complete = "".join(full_response)
            conversation_history.append({"user": query, "assistant": complete})
            if len(conversation_history) > cfg.max_conversation_history:
                conversation_history.pop(0)

            yield f"data: {json.dumps({'done': True, 'sources': last_sources})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/status", methods=["GET"])
def status():
    """Check server health and model availability."""
    model_ok = False
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models] if hasattr(models, "models") else []
        model_ok = any(cfg.llm_model.split(":")[0] in m for m in model_names)
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "persona": cfg.persona_name,
        "model": cfg.llm_model,
        "model_available": model_ok,
        "db_chunks": collection.count(),
        "conversation_length": len(conversation_history),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset conversation history."""
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok", "message": "Conversation history cleared."})


@app.route("/api/sources", methods=["GET"])
def sources():
    """Get sources from the last retrieval."""
    return jsonify({"sources": last_sources})


# ──────────────────────────────────────────────
#  Serve frontend files
# ──────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(".", filename)


# ──────────────────────────────────────────────
#  Run server
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🚀 Server starting at http://localhost:5000")
    print(f"   Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
