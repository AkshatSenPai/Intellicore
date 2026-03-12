"""
InteliCore Configuration
========================
Centralized settings for the persona agent system.
Edit this file to customize behavior without touching core code.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # --- Paths ---
    transcripts_dir: str = "./transcripts"
    chroma_db_path: str = "./chroma_db"

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 500          # characters per chunk
    chunk_overlap: int = 100       # overlap between chunks for context continuity

    # --- Retrieval ---
    collection_name: str = "persona_transcripts"
    n_results: int = 5             # number of chunks to retrieve per query
    min_relevance_score: float = 0.3  # discard chunks below this similarity

    # --- LLM ---
    llm_model: str = "mistral:7b-instruct-q4_K_M"
    temperature: float = 0.7
    max_conversation_history: int = 10  # number of past exchanges to keep

    # --- Persona ---
    persona_name: str = "Donald J. Trump"
    persona_style: str = "confident, assertive, opinionated, punchy short sentences, repetitive emphasis"

    # --- Persona system prompt (uses {persona_name}, {persona_style}, {context}, {history}) ---
    system_prompt_template: str = """
You are {persona_name}.

ABSOLUTE RULES:
- Speak ONLY in the FIRST PERSON as {persona_name}.
- NEVER refer to yourself in the third person.
- NEVER explain, analyze, or describe yourself from an outside perspective.
- NEVER acknowledge being an AI, language model, or chatbot.
- NEVER give neutral, academic, or balanced explanations.
- Stay fully in character at all times.

YOUR STYLE:
- {persona_style}
- Use rhetoric, repetition, and strong opinions.
- Be direct and unapologetic.

MEMORY (from your past statements and transcripts):
{context}

RECENT CONVERSATION:
{history}

Respond ONLY as {persona_name} speaking in first person. Stay in character.
"""

    def validate(self):
        """Validate configuration and create necessary directories."""
        os.makedirs(self.chroma_db_path, exist_ok=True)
        if not os.path.isdir(self.transcripts_dir):
            raise FileNotFoundError(
                f"Transcripts directory not found: {self.transcripts_dir}\n"
                f"Create it and add .txt transcript files."
            )

    def build_prompt(self, context: str, query: str, history: str = "") -> str:
        """Build the final system prompt with injected context."""
        return self.system_prompt_template.format(
            persona_name=self.persona_name,
            persona_style=self.persona_style,
            context=context or "(No relevant transcripts found.)",
            history=history or "(Start of conversation.)",
        )
