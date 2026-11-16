"""
Retrieve top-k chunks for a query using FAISS index.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


# Module-level cache for index and chunks
_index: Optional[faiss.Index] = None
_chunks: Optional[List[Dict[str, Any]]] = None
_model: Optional[SentenceTransformer] = None
_config: Optional[Dict[str, Any]] = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_index_and_chunks(config: Dict[str, Any]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load FAISS index and chunk metadata from config paths."""
    index_path = Path(config["index"]["index_path"])
    chunks_path = Path(config["index"]["chunks_path"])
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path} (config key: index.index_path)")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks metadata not found at {chunks_path} (config key: index.chunks_path)")
    
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    return index, chunks


def _get_model(config: Dict[str, Any]) -> SentenceTransformer:
    """Get or create the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(
            config["embedding"]["model_name"],
            device=config["embedding"]["device"],
        )
    return _model


def _get_cached_index_chunks(config: Dict[str, Any]) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Get cached index and chunks, loading on first call."""
    global _index, _chunks
    if _index is None or _chunks is None:
        _index, _chunks = _load_index_and_chunks(config)
    return _index, _chunks


def embed_query(query: str, model: SentenceTransformer, normalize: bool = True) -> np.ndarray:
    """Embed a single query."""
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embedding.astype(np.float32)


def retrieve(
    question: str,
    k: Optional[int] = None,
    config_path: str = "config.yaml",
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Retrieve top-k chunks for a question.
    
    Args:
        question: User question
        k: Number of chunks to retrieve (default from config)
        config_path: Path to config.yaml
    
    Returns:
        Tuple of (results, best_score) where each result dict contains:
        text, scheme, section, url, retrieved, score
    """
    global _config
    if _config is None:
        _config = load_config(config_path)
    
    config = _config
    top_k = k if k is not None else config["retrieval"]["top_k"]
    score_threshold = config["retrieval"]["score_threshold"]
    normalize = config["embedding"]["normalize_embeddings"]
    
    model = _get_model(config)
    index, chunks = _get_cached_index_chunks(config)
    
    # Embed query
    query_embedding = embed_query(question, model, normalize)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    # Filter by threshold and build results
    results = []
    best_score = 0.0
    
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score >= score_threshold:
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            # Keep only required fields
            result = {
                "text": chunk["text"],
                "scheme": chunk["scheme"],
                "slug": chunk["slug"],
                "section": chunk["section"],
                "url": chunk["url"],
                "retrieved": chunk["retrieved"],
                "score": float(score),
            }
            results.append(result)
            best_score = max(best_score, float(score))
    
    return results, best_score


if __name__ == "__main__":
    # Enable UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    
    test_questions = [
        "What is the eligibility for the post matric scholarship?",
        "छात्रवृत्ति के लिए कौन आवेदन कर सकता है?",
        "What is the price of a cricket bat in Mumbai?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")
        
        results, best_score = retrieve(q)
        print(f"Best score: {best_score:.4f}")
        print(f"Retrieved {len(results)} chunks")
        
        if results:
            print("\nTop 3 results:")
            for i, r in enumerate(results[:3], 1):
                print(f"  {i}. {r['scheme']} | {r['section']} | score={r['score']:.4f}")
        else:
            print("  (no results above threshold)")
