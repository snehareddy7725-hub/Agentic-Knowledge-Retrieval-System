"""
Semantic cache for the Agentic RAG system.

Stores previously-answered questions (as embeddings) alongside their
answers, persisted to disk as JSON. On a new question, compares its
embedding against cached questions using cosine similarity — if a
close-enough match is found, the cached answer is returned instantly
without re-running the full agent pipeline (search + LLM call).

This is SEMANTIC caching, not exact-match: "What is her CPI?" and
"What's her CPI score?" can both hit the same cache entry, since
they're compared by meaning (embedding similarity), not exact text.

Usage:
    cache = SemanticCache(cache_path="data/semantic_cache.json")

    query_embedding = dense_embeddings.embed_query(question)
    hit = cache.get(query_embedding)
    if hit:
        return hit["answer"]  # skip the full agent pipeline

    # ... run the full pipeline, get an answer ...
    cache.add(question, query_embedding, answer, sources=[...])
"""

import json
import os
from datetime import datetime


class SemanticCache:
    def __init__(self, cache_path: str = "data/semantic_cache.json", similarity_threshold: float = 0.92):
        """
        Args:
            cache_path: where to persist the cache as JSON.
            similarity_threshold: how close (cosine similarity, 0-1) a
                new question's embedding must be to a cached question's
                embedding to count as a cache hit. Set conservatively
                high (0.92) by default — a wrong cache hit (serving an
                unrelated cached answer) is worse than a cache miss
                (just answering fresh). Tune this based on testing.
        """
        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        else:
            self.entries = []  # list of {question, embedding, answer, sources, cached_at}

    def _save(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query_embedding):
        """
        Returns the best matching cache entry's {"answer", "sources",
        "matched_question", "similarity"} if similarity clears the
        threshold, else None.
        """
        best_score = -1.0
        best_entry = None

        for entry in self.entries:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            return {
                "answer": best_entry["answer"],
                "sources": best_entry.get("sources", []),
                "matched_question": best_entry["question"],
                "similarity": best_score,
            }
        return None

    def add(self, question: str, embedding, answer: str, sources=None):
        """Store a new question/answer pair in the cache."""
        self.entries.append({
            "question": question,
            "embedding": list(embedding),
            "answer": answer,
            "sources": sources or [],
            "cached_at": datetime.utcnow().isoformat() + "Z",
        })
        self._save()

    def clear(self):
        """Wipe the cache entirely (e.g. after re-processing documents)."""
        self.entries = []
        self._save()
