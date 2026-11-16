from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-m3"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load prepared chunks from JSON."""
    with path.open("r", encoding="utf-8") as stream:
        chunks = json.load(stream)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a JSON list of chunks in {path}")
    return chunks


def fingerprint_chunks(chunks: list[dict[str, Any]]) -> str:
    """Fingerprint chunk ids and text in their original order."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk["id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON through a temporary file and atomically replace the target."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def shard_ranges(count: int, shard_count: int) -> list[tuple[int, int]]:
    """Return contiguous, balanced ranges covering all chunks."""
    base, remainder = divmod(count, shard_count)
    ranges = []
    start = 0
    for shard_idx in range(shard_count):
        end = start + base + (shard_idx < remainder)
        ranges.append((start, end))
        start = end
    return ranges


def encode_shard(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Encode one shard, reducing batch size after CUDA OOM."""
    current_batch = batch_size
    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32), current_batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch <= 1:
                raise
            current_batch = max(1, current_batch // 2)
            print(f"CUDA out of memory; retrying this shard with batch size {current_batch}.", flush=True)


def run(args: argparse.Namespace) -> None:
    """Build resumable embedding shards and a FAISS index, or run a benchmark."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. index.py is GPU-only. Fix the CUDA install; do not fall back to CPU."
        )

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {device_name} ({total_vram:.2f} GB VRAM)", flush=True)

    started = time.perf_counter()
    chunks = load_chunks(args.chunks)
    if not chunks:
        raise ValueError("No chunks found to embed.")
    full_fingerprint = fingerprint_chunks(chunks)
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = args.max_seq_length
    fp16 = not args.no_fp16
    if fp16:
        model.half()
    print(f"Model parameter dtype: {next(model.parameters()).dtype}", flush=True)
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.0f} MB", flush=True)

    lengths = [len(model.tokenizer.encode(chunk["text"], add_special_tokens=True)) for chunk in chunks]
    too_long = sum(length > args.max_seq_length for length in lengths)
    print(f"Chunks exceeding max_seq_length ({args.max_seq_length}): {too_long}", flush=True)

    if args.benchmark is not None:
        sample_count = min(args.benchmark, len(chunks))
        if sample_count < 1:
            raise ValueError("--benchmark must be greater than zero.")
        torch.cuda.reset_peak_memory_stats()
        bench_started = time.perf_counter()
        encode_shard(model, [c["text"] for c in chunks[:sample_count]], args.batch_size)
        elapsed = time.perf_counter() - bench_started
        rate = sample_count / elapsed if elapsed else 0.0
        projected = len(chunks) / rate if rate else float("inf")
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Benchmark chunks: {sample_count}")
        print(f"Chunks/second: {rate:.2f}")
        print(f"Peak GPU memory: {peak_mb:.0f} MB")
        print(f"Projected full-file seconds: {projected:.1f}")
        return

    shard_count = args.shards
    if shard_count < 1:
        raise ValueError("--shards must be at least 1.")
    out_dir = args.out
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    meta_path = shards_dir / "meta.json"
    settings = {
        "fingerprint": full_fingerprint,
        "chunk_count": len(chunks),
        "shard_count": shard_count,
        "model_name": MODEL_NAME,
        "max_seq_length": args.max_seq_length,
        "fp16": fp16,
        "batch_size": args.batch_size,
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing != settings:
            raise SystemExit(f"Shard metadata mismatch at {meta_path}; refusing to overwrite existing shards.")
    else:
        save_json_atomic(meta_path, settings)

    ranges = shard_ranges(len(chunks), shard_count)
    completed_seconds = 0.0
    current_batch_size = args.batch_size
    for shard_idx, (start, end) in enumerate(ranges):
        shard_path = shards_dir / f"emb_{shard_idx:03d}.npy"
        if shard_path.exists():
            completed_seconds += 0.0
            print(f"Shard {shard_idx + 1}/{shard_count} already exists; skipping.", flush=True)
            continue
        shard_started = time.perf_counter()
        print(f"Starting shard {shard_idx + 1}/{shard_count} with batch size {current_batch_size}.", flush=True)
        embeddings, current_batch_size = encode_shard(
            model, [c["text"] for c in chunks[start:end]], current_batch_size
        )
        temp_path = shard_path.with_suffix(".tmp.npy")
        np.save(temp_path, embeddings)
        temp_path.replace(shard_path)
        elapsed = time.perf_counter() - shard_started
        completed_seconds += elapsed
        remaining_count = sum(not (shards_dir / f"emb_{i:03d}.npy").exists() for i in range(shard_idx + 1, shard_count))
        remaining_seconds = (elapsed / max(1, end - start)) * sum(b - a for a, b in ranges[shard_idx + 1:])
        print(
            f"Shard {shard_idx + 1}/{shard_count} saved in {elapsed:.1f}s; "
            f"estimated remaining {remaining_seconds:.1f}s ({remaining_count} shards).",
            flush=True,
        )

    vectors = [np.load(shards_dir / f"emb_{i:03d}.npy", mmap_mode="r") for i in range(shard_count)]
    embeddings = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(f"Embedding row count {embeddings.shape[0]} != chunk count {len(chunks)}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with (out_dir / "chunks.json").open("w", encoding="utf-8") as stream:
        json.dump(chunks, stream, ensure_ascii=False, indent=1)
    build_info = {
        "num_chunks": len(chunks),
        "model_name": MODEL_NAME,
        "max_seq_length": args.max_seq_length,
        "device": "cuda",
        "fp16": fp16,
        "shards": shard_count,
        "build_time_seconds": time.perf_counter() - started,
        "input_fingerprint": full_fingerprint,
    }
    save_json_atomic(out_dir / "build_info.json", build_info)
    print(f"Index built with {len(chunks)} vectors in {build_info['build_time_seconds']:.1f}s.", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GPU-only resumable FAISS index builder")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    parser.add_argument("--out", type=Path, default=Path("data/index"))
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--benchmark", type=int, metavar="N")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted. Completed shards are saved; rerun the same command to resume.", flush=True)
