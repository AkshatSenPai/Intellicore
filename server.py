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
import uuid
import json
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context, session
from flask_cors import CORS

import ollama
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from config import Config

# ──────────────────────────────────────────────
#  Initialize
# ──────────────────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = os.environ.get("INTELLICORE_SECRET_KEY", os.urandom(24))

cfg = Config()

allowed_origins_env = os.environ.get("INTELLICORE_ALLOWED_ORIGINS", "").strip()
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = cfg.allowed_origins

CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=True)

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

# --- Session state (per-session, in-memory) ---
session_store = {}


# ──────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────

def get_session_state():
    """Get or create per-session state."""
    session_id = session.get("sid")
    if not session_id:
        session_id = uuid.uuid4().hex
        session["sid"] = session_id

    state = session_store.get(session_id)
    if not state:
        state = {
            "history": [],
            "last_sources": [],
            "credits": cfg.credit_limit,
        }
        session_store[session_id] = state

    return session_id, state

def retrieve_context(query: str):
    """Retrieve relevant transcript chunks for a query."""
    doc_count = collection.count()
    if doc_count == 0:
        return "", []

    q_embed = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embed,
        n_results=min(cfg.n_results, doc_count),
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

    return "\n\n---\n\n".join(chunks) if chunks else "", sorted(sources)


def format_history(history):
    """Format recent conversation history for the prompt."""
    if not history:
        return ""
    lines = []
    history_limit = max(1, cfg.max_conversation_history)
    for h in history[-history_limit:]:
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

    _session_id, state = get_session_state()

    if state["credits"] <= 0:
        return jsonify({
            "error": "No credits remaining",
            "remaining_credits": 0,
            "credit_limit": cfg.credit_limit,
        }), 402

    try:
        # Retrieve context from ChromaDB
        context, sources = retrieve_context(query)
        state["last_sources"] = sources

        # Build prompt
        history_str = format_history(state["history"])
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
        state["history"].append({"user": query, "assistant": reply})
        if len(state["history"]) > cfg.max_conversation_history:
            state["history"].pop(0)

        state["credits"] -= 1

        return jsonify({
            "response": reply,
            "sources": sources,
            "credits_used": 1,
            "remaining_credits": state["credits"],
            "credit_limit": cfg.credit_limit,
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

    _session_id, state = get_session_state()

    if state["credits"] <= 0:
        return jsonify({
            "error": "No credits remaining",
            "remaining_credits": 0,
            "credit_limit": cfg.credit_limit,
        }), 402

    # Retrieve context
    context, sources = retrieve_context(query)
    state["last_sources"] = sources
    history_str = format_history(state["history"])
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
            state["history"].append({"user": query, "assistant": complete})
            if len(state["history"]) > cfg.max_conversation_history:
                state["history"].pop(0)

            state["credits"] -= 1

            yield f"data: {json.dumps({'done': True, 'sources': sources, 'remaining_credits': state['credits'], 'credit_limit': cfg.credit_limit})}\n\n"

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
    _session_id, state = get_session_state()
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
        "conversation_length": len(state["history"]),
        "remaining_credits": state["credits"],
        "credit_limit": cfg.credit_limit,
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset conversation history."""
    _session_id, state = get_session_state()
    state["history"] = []
    state["last_sources"] = []
    return jsonify({
        "status": "ok",
        "message": "Conversation history cleared.",
        "remaining_credits": state["credits"],
        "credit_limit": cfg.credit_limit,
    })


@app.route("/api/sources", methods=["GET"])
def sources():
    """Get sources from the last retrieval."""
    _session_id, state = get_session_state()
    return jsonify({"sources": state["last_sources"]})


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
