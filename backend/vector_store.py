import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

from backend.config import (
    BOOKS_DIR,
    EMBEDDING_MODEL_NAME,
    ONNX_THREADS,
    TOP_K_RETRIEVAL,
    RERANK_TOP_K
)

# Global singleton for FastEmbed model to avoid repeated ONNX initialization
_EMBED_MODEL = None

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        # Initialized with ONNX runtime configured for CPU threads
        _EMBED_MODEL = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            threads=ONNX_THREADS
        )
    return _EMBED_MODEL

class BookVectorStore:
    """
    Manages isolated RAG knowledge base for each uploaded book:
    - Vector embeddings (FastEmbed ONNX CPU)
    - BM25 lexical index (Rank-BM25)
    - Hybrid search with Reciprocal Rank Fusion (RRF) reranking
    """

    def __init__(self, book_id: str):
        self.book_id = book_id
        self.book_dir = BOOKS_DIR / book_id
        self.book_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.book_dir / "metadata.json"
        self.chunks_path = self.book_dir / "chunks.json"
        self.embeddings_path = self.book_dir / "embeddings.npy"
        self.bm25_path = self.book_dir / "bm25.pkl"

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: Dict[str, Any] = {}

        if self.is_ready():
            self.load()

    def is_ready(self) -> bool:
        """Checks if the book knowledge base has been built and saved."""
        return (
            self.metadata_path.exists() and
            self.chunks_path.exists() and
            self.embeddings_path.exists()
        )

    def build_knowledge_base(self, metadata: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        """
        Builds vector embeddings and BM25 index for the book chunks and persists them.
        """
        self.metadata = metadata
        self.chunks = chunks

        # 1. Generate embeddings using FastEmbed
        texts_to_embed = [c.get("enriched_content", c["content"]) for c in chunks]
        embed_model = get_embed_model()

        # Generator to numpy array
        vectors_gen = embed_model.embed(texts_to_embed, batch_size=32)
        embeddings_list = list(vectors_gen)
        self.embeddings = np.array(embeddings_list, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = self.embeddings / norms

        # 2. Build BM25 Index
        tokenized_corpus = [
            re_tokenize(c.get("enriched_content", c["content"]))
            for c in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Save to disk
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        np.save(str(self.embeddings_path), self.embeddings)

        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self) -> None:
        """Loads stored chunks, embeddings, and BM25 index from disk."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embeddings = np.load(str(self.embeddings_path))

        if self.bm25_path.exists():
            with open(self.bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            tokenized_corpus = [
                re_tokenize(c.get("enriched_content", c["content"]))
                for c in self.chunks
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL, rerank_k: int = RERANK_TOP_K) -> List[Dict[str, Any]]:
        """
        Hybrid search combining:
        1. Dense Vector Similarity (FastEmbed Cosine)
        2. Lexical BM25 Search
        3. Reciprocal Rank Fusion (RRF) Reranking
        """
        if not self.chunks or self.embeddings is None:
            return []

        n_chunks = len(self.chunks)
        fetch_k = min(top_k * 2, n_chunks)

        # --- 1. Dense Vector Retrieval ---
        embed_model = get_embed_model()
        q_gen = embed_model.embed([query])
        q_vec = np.array(list(q_gen)[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarities
        dense_scores = np.dot(self.embeddings, q_vec)
        top_dense_indices = np.argsort(dense_scores)[::-1][:fetch_k]

        dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(top_dense_indices)}

        # --- 2. BM25 Lexical Retrieval ---
        q_tokens = re_tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(top_bm25_indices)}

        # --- 3. Reciprocal Rank Fusion (RRF) Reranking ---
        # RRF formula: RRF_score(d) = \sum_{m} \frac{1}{60 + rank_m(d)}
        combined_candidate_indices = set(top_dense_indices) | set(top_bm25_indices)
        rrf_scores = {}

        for idx in combined_candidate_indices:
            score = 0.0
            if idx in dense_rank_map:
                score += 1.0 / (60.0 + dense_rank_map[idx])
            if idx in bm25_rank_map:
                score += 1.0 / (60.0 + bm25_rank_map[idx])
            rrf_scores[idx] = score

        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:rerank_k]

        results = []
        for rank, idx in enumerate(sorted_indices):
            chunk = self.chunks[idx].copy()
            chunk["retrieval_rank"] = rank + 1
            chunk["rrf_score"] = round(float(rrf_scores[idx]), 4)
            chunk["dense_score"] = round(float(dense_scores[idx]), 4)
            chunk["bm25_score"] = round(float(bm25_scores[idx]), 4)
            results.append(chunk)

        return results

def re_tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 lexical indexing."""
    import re
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text.lower())
    return tokens
