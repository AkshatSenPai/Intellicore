"""
InteliCore — Persona Agent (Interactive Chat)
===============================================
RAG-powered persona chatbot with conversation memory,
streaming responses, and improved retrieval.

Improvements over original:
  - Conversation history maintained across turns
  - Streaming output for responsive UX
  - Relevance filtering (discards low-quality retrievals)
  - Source attribution (shows which transcripts were used)
  - Graceful error handling and recovery
  - Configurable via config.py
  - Special commands (/reset, /sources, /config, /help)
"""

import sys
import os
from typing import List, Tuple, Optional

import ollama
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from config import Config


# ──────────────────────────────────────────────
#  Retrieval
# ──────────────────────────────────────────────

class Retriever:
    """Handles embedding queries and retrieving relevant chunks from ChromaDB."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = PersistentClient(path=os.path.abspath(cfg.chroma_db_path))
        self.collection = self.client.get_or_create_collection(cfg.collection_name)
        self.embedder = SentenceTransformer(cfg.embedding_model)

        doc_count = self.collection.count()
        if doc_count == 0:
            print("⚠️  Warning: ChromaDB collection is empty!")
            print("   Run `python build_chroma.py` first to index transcripts.\n")
        else:
            print(f"📚 Loaded {doc_count} chunks from vector database.")

    def query(self, text: str) -> Tuple[str, List[str]]:
        """
        Retrieve relevant transcript chunks for a query.
        Returns (combined_context, list_of_source_filenames).
        """
        if self.collection.count() == 0:
            return "", []

        q_embed = self.embedder.encode([text]).tolist()
        results = self.collection.query(
            query_embeddings=q_embed,
            n_results=min(self.cfg.n_results, self.collection.count()),
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
                # ChromaDB returns L2 distance; lower = more similar
                # Convert to a rough similarity score for filtering
                similarity = 1 / (1 + dist)
                if similarity >= self.cfg.min_relevance_score:
                    chunks.append(doc)
                    if meta and "source" in meta:
                        sources.add(meta["source"])

        context = "\n\n---\n\n".join(chunks) if chunks else ""
        return context, sorted(sources)


# ──────────────────────────────────────────────
#  Conversation Manager
# ──────────────────────────────────────────────

class ConversationManager:
    """Maintains conversation history for multi-turn context."""

    def __init__(self, max_turns: int = 10):
        self.history: List[dict] = []
        self.max_turns = max_turns

    def add_exchange(self, user_msg: str, assistant_msg: str):
        self.history.append({"user": user_msg, "assistant": assistant_msg})
        # Trim oldest exchanges if over limit
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def format_history(self) -> str:
        if not self.history:
            return ""
        lines = []
        for h in self.history[-self.max_turns:]:
            lines.append(f"User: {h['user']}")
            lines.append(f"You: {h['assistant']}")
        return "\n".join(lines)

    def reset(self):
        self.history.clear()


# ──────────────────────────────────────────────
#  Agent
# ──────────────────────────────────────────────

class PersonaAgent:
    """Main agent combining retrieval, conversation memory, and LLM generation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.retriever = Retriever(cfg)
        self.conversation = ConversationManager(cfg.max_conversation_history)
        self._last_sources: List[str] = []

        # Verify Ollama model is available
        self._check_model()

    def _check_model(self):
        """Check if the configured Ollama model is available."""
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models] if hasattr(models, 'models') else []
            # Also check short names (without tag)
            short_names = [n.split(":")[0] for n in model_names]
            target_short = self.cfg.llm_model.split(":")[0]

            if self.cfg.llm_model not in model_names and target_short not in short_names:
                print(f"⚠️  Model '{self.cfg.llm_model}' not found locally.")
                print(f"   Available: {', '.join(model_names) or '(none)'}")
                print(f"   Run: ollama pull {self.cfg.llm_model}\n")
        except Exception as e:
            print(f"⚠️  Could not connect to Ollama: {e}")
            print("   Make sure Ollama is running (ollama serve).\n")

    def respond(self, query: str, stream: bool = True) -> str:
        """Generate a persona response for the given query."""

        # Retrieve context
        context, sources = self.retriever.query(query)
        self._last_sources = sources

        # Build prompt
        history_str = self.conversation.format_history()
        system_prompt = self.cfg.build_prompt(context, query, history_str)

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Generate with streaming
        if stream:
            response_text = self._stream_response(messages)
        else:
            response = ollama.chat(
                model=self.cfg.llm_model,
                messages=messages,
                options={"temperature": self.cfg.temperature},
            )
            response_text = response["message"]["content"]

        # Save to conversation history
        self.conversation.add_exchange(query, response_text)

        return response_text

    def _stream_response(self, messages: list) -> str:
        """Stream response token by token for better UX."""
        full_response = []
        try:
            stream = ollama.chat(
                model=self.cfg.llm_model,
                messages=messages,
                options={"temperature": self.cfg.temperature},
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_response.append(token)
            print()  # newline after streaming completes
        except Exception as e:
            print(f"\n❌ Generation error: {e}")
            return "(Error generating response.)"

        return "".join(full_response)

    def get_last_sources(self) -> List[str]:
        return self._last_sources


# ──────────────────────────────────────────────
#  Interactive CLI
# ──────────────────────────────────────────────

HELP_TEXT = """
╔══════════════════════════════════════╗
║        InteliCore Commands           ║
╠══════════════════════════════════════╣
║  /help     — Show this help          ║
║  /sources  — Show last retrieval     ║
║  /reset    — Clear conversation      ║
║  /config   — Show current settings   ║
║  exit      — Quit                    ║
╚══════════════════════════════════════╝
"""


def main():
    cfg = Config()

    print("\n" + "=" * 50)
    print("  🧠 InteliCore — AI Persona Agent")
    print(f"  Persona: {cfg.persona_name}")
    print(f"  Model:   {cfg.llm_model}")
    print("=" * 50)

    try:
        cfg.validate()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    agent = PersonaAgent(cfg)
    print(f"\nType your message or /help for commands.\n")

    while True:
        try:
            query = input(f"🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue

        # --- Commands ---
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        if query.lower() == "/help":
            print(HELP_TEXT)
            continue

        if query.lower() == "/reset":
            agent.conversation.reset()
            print("🔄 Conversation history cleared.\n")
            continue

        if query.lower() == "/sources":
            sources = agent.get_last_sources()
            if sources:
                print(f"📂 Sources used in last response:")
                for s in sources:
                    print(f"   • {s}")
            else:
                print("📂 No sources retrieved yet.")
            print()
            continue

        if query.lower() == "/config":
            print(f"  Persona:     {cfg.persona_name}")
            print(f"  Model:       {cfg.llm_model}")
            print(f"  Temperature: {cfg.temperature}")
            print(f"  Chunk size:  {cfg.chunk_size}")
            print(f"  Retrieval:   top {cfg.n_results} chunks")
            print(f"  History:     last {cfg.max_conversation_history} turns")
            print(f"  DB path:     {os.path.abspath(cfg.chroma_db_path)}")
            print()
            continue

        # --- Generate response ---
        print(f"\n🤖 {cfg.persona_name}: ", end="", flush=True)

        try:
            agent.respond(query, stream=True)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Make sure Ollama is running and the model is downloaded.\n")
            continue

        # Show sources hint
        sources = agent.get_last_sources()
        if sources:
            print(f"   📂 [{len(sources)} source(s) — type /sources for details]")
        print()


if __name__ == "__main__":
    main()
