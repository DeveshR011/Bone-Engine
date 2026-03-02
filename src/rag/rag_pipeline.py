"""
RAG Integration Pipeline
========================
Query → Hybrid Retrieval → Top-K docs → Prompt Construction → LLM → Answer

Supports OpenAI and HuggingFace backends.  Falls back to prompt-only mode
(no LLM call) when no API key / model is available.
"""

from __future__ import annotations

import os
from typing import Any

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)

# Default system prompt template
PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        retriever: Any = None,
        llm_backend: str = "openai",
        openai_model: str = "gpt-3.5-turbo",
        hf_model: str = "google/flan-t5-base",
        top_k: int = 5,
        max_context_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        """
        Args:
            retriever: A callable/object with a `search(query) -> list[dict]` method.
                       Each dict must contain at least ``doc_id`` and ``content``.
            llm_backend: "openai" or "huggingface".
            top_k: Number of documents to retrieve for context.
        """
        self.retriever = retriever
        self.llm_backend = llm_backend
        self.openai_model = openai_model
        self.hf_model_name = hf_model
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature

        self._hf_pipeline = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        retrieval_fn: Any = None,
    ) -> dict[str, Any]:
        """Run the full RAG pipeline.

        Args:
            question: User question.
            retrieval_fn: Optional override — callable(question) → list[dict].
                          If not provided, uses self.retriever.

        Returns dict with keys:
            - question
            - retrieved_docs: list of {doc_id, content, score}
            - context: assembled context string
            - answer: LLM-generated answer (or prompt-only if no LLM)
            - retrieval_latency_ms
            - generation_latency_ms
        """
        # --- Retrieval ---
        with Timer("rag-retrieval") as t_ret:
            if retrieval_fn is not None:
                docs = retrieval_fn(question)
            elif self.retriever is not None:
                docs = self.retriever(question)
            else:
                docs = []

        docs = docs[: self.top_k]
        retrieval_ms = t_ret.elapsed_ms

        # --- Construct context ---
        context = self._build_context(docs)

        # --- Generate ---
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        with Timer("rag-generation") as t_gen:
            answer_text = self._generate(prompt)
        generation_ms = t_gen.elapsed_ms

        result = {
            "question": question,
            "retrieved_docs": [
                {"doc_id": d.get("doc_id", ""), "content": d.get("content", "")[:200], "score": d.get("score", 0)}
                for d in docs
            ],
            "context": context[:500] + ("…" if len(context) > 500 else ""),
            "answer": answer_text,
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
        }

        logger.info(
            "RAG: retrieved %d docs (%.1f ms), generated answer (%.1f ms)",
            len(docs),
            retrieval_ms,
            generation_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_context(self, docs: list[dict[str, Any]]) -> str:
        """Assemble top-K document contents into a context string."""
        parts = []
        char_budget = self.max_context_tokens * 4  # rough chars-per-token estimate
        used = 0
        for i, doc in enumerate(docs, 1):
            text = doc.get("content", "").strip()
            if used + len(text) > char_budget:
                text = text[: char_budget - used]
            parts.append(f"[Document {i}]  {text}")
            used += len(text)
            if used >= char_budget:
                break
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Generation backends
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Dispatch to the configured LLM backend."""
        if self.llm_backend == "openai":
            return self._generate_openai(prompt)
        elif self.llm_backend == "huggingface":
            return self._generate_hf(prompt)
        else:
            # Fallback: return prompt only
            return f"[NO LLM] Prompt:\n{prompt}"

    def _generate_openai(self, prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — returning prompt-only output")
            return f"[NO LLM — set OPENAI_API_KEY] Prompt:\n{prompt}"

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            return f"[OpenAI error: {e}]"

    def _generate_hf(self, prompt: str) -> str:
        try:
            if self._hf_pipeline is None:
                from transformers import pipeline as hf_pipeline

                self._hf_pipeline = hf_pipeline(
                    "text2text-generation",
                    model=self.hf_model_name,
                    max_new_tokens=256,
                )
            result = self._hf_pipeline(prompt)
            return result[0]["generated_text"].strip()
        except Exception as e:
            logger.error("HuggingFace generation failed: %s", e)
            return f"[HF error: {e}]"
