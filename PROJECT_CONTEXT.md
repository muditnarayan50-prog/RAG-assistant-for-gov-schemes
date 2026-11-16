# Project: Local Multilingual RAG Assistant for Government Schemes

## Goal

Answer questions in Hindi or English about Indian government schemes, using only
retrieved official text, with citations (scheme, section, URL, retrieval date).
Abstain with "not found" when retrieval confidence is low.

## Pipeline (files)

1. prepare_chunks.py: myScheme JSONL -> clean section-level chunks
2. index.py: embeds chunks with BAAI/bge-m3 (sentence-transformers), builds FAISS index
3. retrieve.py: embeds the question, returns top-k chunks and the best score
4. llm.py: provider switch. local = Ollama (qwen3:4b, CPU only), groq = Groq (opt-in)
5. answer.py: retrieve -> threshold check in code -> prompt -> generate -> citations
6. app.py: Streamlit UI with provider toggle and a cloud-privacy notice
7. evaluate.py: Recall@5, MRR, abstention, latency, faithfulness, no-retrieval baseline

## Constraints

- Python 3.13 on Windows; must run on a low-end laptop with no GPU (build the index on GPU, run on CPU)
- Local mode makes no network calls at runtime (set HF_HUB_OFFLINE=1 after models are downloaded)
- Groq is opt-in only; the UI must say the question is sent to the cloud
- All settings live in config.yaml, not in code
- API keys come from environment variables only; never hard-code keys
- Never commit the full myScheme JSONL or data/raw to git
- pathlib only; type hints; short docstrings; keep functions small
- Qwen3 thinking must be off (think=False), answers capped near 300 tokens

## Scope (2-week plan)

- 10 schemes, about 30 evaluation questions (some unanswerable), dense retrieval only
- One local model (qwen3:4b) and one Groq model; no BM25, no ablations

## Status

- Done and tested: prepare_chunks.py (on a small synthetic record), index.py (200-chunk test index), retrieve.py (3 test questions)
- Written, not fully verified: llm.py, answer.py
- Not started or not run: 10-scheme index, app.py, evaluate.py
