# Local Multilingual RAG Assistant for Government Schemes — Project Log

## Overview
Built a complete RAG pipeline to answer questions about Indian government schemes in English/Hindi using only official text from myScheme.gov.in, with citations and abstention when confidence is low.

---

## Files Created

| File | Purpose |
|------|---------|
| `prepare_chunks.py` | Clean JSONL → section-level chunks (already existed) |
| `config.yaml` | All configuration (no hardcoded values in code) |
| `index.py` | Embed chunks with BAAI/bge-m3 → FAISS index |
| `retrieve.py` | Query embedding + FAISS search → top-k chunks |
| `llm.py` | Provider abstraction: Ollama (local) + Groq (cloud) |
| `answer.py` | Pipeline: retrieve → threshold → prompt → generate → format citations |
| `app.py` | Streamlit UI with provider toggle + cloud warning |
| `evaluate.py` | Recall@5, MRR, abstention rate, latency |
| `requirements.txt` | Python dependencies |
| `PROJECT_LOG.md` | This file |
| `ERROR_LOG.txt` | Error tracking |

---

## Pipeline Flow

```
User Question
     │
     ▼
retrieve.py ──embed query──▶ FAISS search ──▶ top-k chunks
     │                              │
     │                              ▼
     │                    score ≥ threshold?
     │                              │
     ├────────────── No ────────────┤
     │                              │
     ▼                              ▼
"not found"              answer.py ──▶ LLM (Ollama/Groq)
                              │
                              ▼
                    Format with citations
                              │
                              ▼
                         Display to user
```

---

## Embedding Part — Deep Dive

### Why BAAI/bge-m3?
- **Multilingual**: Handles English + Hindi (and 100+ languages) natively
- **Strong retrieval**: SOTA on MTEB retrieval benchmarks
- **1024-dim embeddings**: Good quality/size tradeoff
- **CPU-friendly**: Runs on laptop without GPU

### Key Settings in `index.py`

```python
model = SentenceTransformer("BAAI/bge-m3", device=device)
model.max_seq_length = 512        # ← Critical for speed
```

| Setting | Value | Why |
|---------|-------|-----|
| `max_seq_length` | 512 | Default is 8192. Most scheme chunks are <512 tokens. 16× faster encoding, 16× less memory. |
| `batch_size` | 16 | Fits in 16 GB RAM. Larger = OOM. |
| `normalize_embeddings` | True | Enables cosine similarity via FAISS `IndexFlatIP` (inner product). |
| `device` | auto (cuda/cpu) | Uses GPU if available, else CPU. |

### Checkpointing (Crash-Resistant)
```python
# Every 50 batches:
save_checkpoint(out_dir, embeddings, batch_idx, total_batches)
```
- Saves `embeddings_checkpoint.npy` + `checkpoint_meta.json`
- On restart, loads partial embeddings and resumes from last batch
- Cleanup on success

### Subsampling for Fast Iteration
```bash
--max-chunks 500  # Random sample with seed=42
```
- 500 chunks → ~10 min on CPU
- Full 85,751 chunks → ~28 hours on CPU
- Use subset for development, full for production

### Output Files
```
data/index/
├── faiss.index          # FAISS IndexFlatIP (inner product = cosine)
├── chunks.json          # Chunks in same order as index vectors
└── build_info.json      # {num_chunks, model, max_seq_length, device, build_time_sec}
```

---

## Configuration (`config.yaml`)

All tunable params in one place:
- Embedding model, device, batch size
- FAISS index type (flat/ivf/hnsw)
- Retrieval: top_k, score_threshold, max_context_chars
- LLM: local Ollama model, temperature, `num_predict`, `num_ctx`, `num_gpu`, `num_thread`, `think`; Groq model, temperature, `max_tokens`
- UI: languages, citation display
- Evaluation: recall@k, MRR

---

## Running the Pipeline

### 1. Build Index (done)
```bash
python index.py --chunks data/chunks.json --out data/index --max-chunks 500 --batch-size 16 --max-seq-length 512
```

### 2. Update Config
Edit `config.yaml`:
```yaml
index:
  index_path: "data/index/faiss.index"
  chunks_path: "data/index/chunks.json"
```

### 3. Run App (Local - Ollama)
```bash
# Start Ollama first: ollama serve
# Pull model: ollama pull qwen3:4b
streamlit run app.py
```

### 4. Run App (Cloud - Groq)
```bash
$env:GROQ_API_KEY = "your-key"
# In UI, select "Cloud (Groq)" provider
streamlit run app.py
```

### 5. Evaluate
```bash
python evaluate.py
```

---

## Performance Notes

| Dataset Size | Device | Time (est.) |
|--------------|--------|-------------|
| 500 chunks | CPU | 10 min |
| 5,000 chunks | CPU | ~1.5 hr |
| 85,751 chunks | CPU | ~28 hr |
| 85,751 chunks | GPU (RTX 3080) | ~45 min |

**Tips for speed:**
- Reduce `max_seq_length` to 256 if chunks are short
- Use `--max-chunks` for development
- Use GPU if available (auto-detected)
- Consider `IndexIVFFlat` or `IndexHNSWFlat` for larger datasets (faster search, slight recall tradeoff)

---

## Abstention Logic

The system says "not found" / "नहीं मिला" when:
1. **Retrieval score < threshold** (default 0.35 cosine similarity)
2. **LLM responds with "not found"** (even if retrieval passed)

This prevents hallucination on low-confidence queries.

---

## Progress Log

### 2026-11-13 — Completed Steps

1. ✅ **Rebuilt index** with 200 chunks (seed=42) — 2.5 min on CPU
   - Output: `data/index/{faiss.index, chunks.json, build_info.json}`

2. ✅ **Fixed `retrieve.py`** for Windows UTF-8 support
   - Loads paths from `config.yaml` (index.index_path, index.chunks_path)
   - Module-level caching of index, chunks, model
   - `sys.stdout.reconfigure(encoding="utf-8")` for Hindi/Devanagari output
   - Public `retrieve(question, k=None)` function with proper return format

3. ✅ **Verification queries** (all working):
   - English: "What is the eligibility for the post matric scholarship?" → score 0.6239
   - Hindi: "छात्रवृत्ति के लिए कौन आवेदन कर सकता है?" → score 0.6575
   - Unanswerable: "What is the price of a cricket bat in Mumbai?" → score 0.4000

4. ✅ **Installed Ollama** and pulled `qwen3:4b` (2.5 GB)
   - Updated `config.yaml` local model to `qwen3:4b`
   - Ollama server responding locally; `llm.py` now uses the native Python client instead of the `/v1` endpoint

5. ✅ **Updated `answer.py`** to use new `retrieve()` function
   - Removed dependency on `Retriever` class
   - Uses `retrieve()` directly with config path

### 2026-11-14 — Answer Pipeline Verification

1. ✅ **Tested `answer.py` with the three verification questions** using local Ollama (`qwen3:4b`)
   - English post-matric scholarship question: answer generated; best score 0.6239
   - Hindi scholarship question: answer generated; best score 0.6575
   - Unanswerable cricket-bat price question: abstained with "not found"; best score 0.4000

2. ✅ **Switched local generation to native Ollama and configured CPU settings**
   - `config.yaml`: `num_predict: 300`, `num_ctx: 4096`, `num_gpu: 0`, `num_thread: 4`, `think: false`
   - `llm.py` calls `ollama.chat`, removes `<think>...</think>` blocks, raises on empty output, and logs elapsed time when `LOG_LEVEL=debug`
   - Groq remains on the OpenAI-compatible provider with its existing configuration
   - `python -m py_compile llm.py`, mocked Ollama option/cleanup/logging checks, and Pylance diagnostics passed
   - The `ollama` Python package was installed in the current environment; `requirements.txt` was not changed

3. ✅ **Measured Hindi inference on CPU** with `LOG_LEVEL=debug` and `HF_HUB_OFFLINE=1`
   - `ollama ps` showed `qwen3:4b` using 100% CPU, 3.2 GB, and a 4096-token context
   - Ollama generation took 76.12 seconds; retrieval best score was 0.6575
   - The output hit the 300-token limit and included untagged reasoning, so the `<think>` tag filter did not remove it and the answer was truncated

4. ✅ **Handled Qwen3 reasoning and token-limit responses in `llm.py`**
   - Keeps text after the last `</think>`, removes remaining tagged blocks, then trims whitespace
   - Raises a specific error when a length-limited response is empty or starts with recognizable reasoning; also raises on any other empty cleaned response
   - Added `python llm.py --selftest` with mocked cases for closing-tag-only, paired tags, plain text, empty cleaned output, and token-exhausted reasoning
   - `python -m py_compile llm.py` and `python llm.py --selftest` passed; Groq behavior remains unchanged

### Next Steps (In Progress)

- [x] Test `answer.py` with the three verification questions
- [ ] Run `app.py` with a local question
- [ ] Write `evaluate.py` with baseline and faithfulness metrics

---

## Next Steps

- [ ] Fix detailed-mode cross-scheme contamination: a single scheme named explicitly in the question (for example PM-KISAN) must take priority over score-gap ambiguity; reserve multi-scheme context for questions that actually ask about multiple schemes or remain genuinely ambiguous.
- [ ] Add a regression check that verifies detailed context and cited source slugs stay within the explicitly named scheme.
- [ ] Build the intended full scheme index and rerun retrieval evaluation before threshold tuning.

---

## Progress Update — 2026-11-16

### Configuration and generation

- Local Ollama is configured for `qwen3:4b-instruct-2507-q4_K_M`. `think: null` makes `llm.py` omit the `think` argument for this non-thinking build; `python -m py_compile llm.py` and `python llm.py --selftest` passed. No model was run during that code change.
- Groq remains opt-in with `openai/gpt-oss-20b`, `reasoning_effort: low`, and `max_tokens: 2048`. The provider requests hidden reasoning and raises an error for an empty response stopped by the token limit. The configuration change was syntax-checked; no live Groq request was made.
- Added brief and detailed answer settings. Brief is the default; detailed answers can expand scheme sections, use a larger context, and request more local output tokens. `answer.generate()` accepts `detail="brief"` or `detail="detailed"`.
- Retrieval is configured for `top_k: 8`; `max_context_chars` remains 4000 for brief mode.
- `README.md` documents the goal, architecture, flow chart, setup, and run commands.

### Indexing

- Reworked `index.py` for GPU-only indexing with resumable shards, a fingerprint check, FP16 model execution by default, OOM batch-size reduction, and a benchmark option. `python -m py_compile index.py` passed.
- The GPU benchmark did not run. The isolated `.venv-gpu` environment had a persistent `PermissionError` reading `colorama/__init__.py`, including after reinstalling that package. No full embedding job was started.

### Answer grounding and latest observed issue

- `answer.py` now filters the Sources list to valid passage numbers cited by the answer, includes level/state metadata where available, and distinguishes low-score abstention from model “not found”. Its detailed mode expands sections for the top scheme and uses `focus_gap: 0.02` to detect close-scoring schemes. `detailed.expand_top_schemes` is 1.
- On a user-run detailed query, “What are the benefits and eligibility of PM-KISAN?”, the answer included Namo Shetkari facts and cited its passage as support. The second user-run query about Atal Pension Yojana did not provide a source citation and returned a mixed explanation followed by “not found”. These are observed model outputs, not results from a new agent-run model test.
- Likely cause for the first failure: the score-gap check classified the PM-KISAN question as ambiguous and retained multiple schemes. Scores were not included in the supplied output, so this cause is unconfirmed.
- Proposed next change: when a question uniquely names one scheme, prioritize that scheme and keep detailed context and citations on its slug despite a close score from another scheme. Use multi-scheme context only when the question asks about multiple schemes or no single scheme can be resolved. This fix is not yet implemented or verified.

### Recent checks and remaining work

- The YAML parser error reported while starting Streamlit was addressed by saving `config.yaml` as UTF-8 without a BOM; a subsequent `yaml.safe_load` succeeded and confirmed `top_k: 8` and `max_context_chars: 4000`.
- A fresh Streamlit instance previously returned HTTP 200 at `http://localhost:8503`; that only verified page loading, not question answering.
- The project still uses a 200-chunk development index. Building the full index, evaluating answer quality, and fixing the detailed-mode scheme-focus issue remain outstanding.
