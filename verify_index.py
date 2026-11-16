"""Read-only check of the built index. Prints results; changes no files.
Run from the project folder:  python verify_index.py
"""
import json, sys
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

idx = faiss.read_index("data/index/faiss.index")
chunks = json.load(open("data/index/chunks.json", encoding="utf8"))
print("ntotal", idx.ntotal, "dim", idx.d, "chunks", len(chunks), "match", idx.ntotal == len(chunks))
print(json.load(open("data/index/build_info.json")))

qs = [
    "What is the eligibility for the post matric scholarship?",
    "छात्रवृत्ति के लिए कौन आवेदन कर सकता है?",
    "What documents are needed for Ayushman Bharat?",
    "What is the price of a cricket bat in Mumbai?",
]

cpu = SentenceTransformer("BAAI/bge-m3", device="cpu")
cpu.max_seq_length = 512
q_cpu = cpu.encode(qs, normalize_embeddings=True).astype("float32")

gpu = SentenceTransformer("BAAI/bge-m3", device="cuda")
gpu.max_seq_length = 512
gpu.half()
q_gpu = gpu.encode(qs, normalize_embeddings=True).astype("float32")
print("max abs diff, GPU vs CPU query vectors:", float(np.abs(q_cpu - q_gpu).max()))

s_cpu, i_cpu = idx.search(q_cpu, 5)
_, i_gpu = idx.search(q_gpu, 5)
for n, q in enumerate(qs):
    print("\n" + q, "| best score", round(float(s_cpu[n][0]), 4),
          "| top-5 overlap CPU vs GPU:", len(set(i_cpu[n]) & set(i_gpu[n])), "/ 5")
    for r in range(5):
        c = chunks[i_cpu[n][r]]
        print(" ", r + 1, round(float(s_cpu[n][r]), 4), c["scheme"][:60], "|", c["section"])
