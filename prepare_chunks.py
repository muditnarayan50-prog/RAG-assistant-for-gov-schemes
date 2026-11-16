"""
Turn the myScheme JSONL into clean, section-level chunks for RAG.

Usage examples:
  python prepare_chunks.py schemes.jsonl --stats
  python prepare_chunks.py schemes.jsonl --categories Education --level Central --limit 6 \
         --out data/subset_a
  python prepare_chunks.py schemes.jsonl --slugs-file my_slugs.txt --exclude-faq-from-index

Outputs (in --out folder):
  chunks.json           one dict per chunk: id, text, scheme, slug, section, url, references, ...
  schemes_clean.json    cleaned scheme records (no rich-text trees)
  eval_candidates.json  every FAQ (question + official answer) of the chosen schemes
"""
import argparse, html, json, random, re
from pathlib import Path

MAX_CHARS = 1200

def clean(s):
    if not s:
        return ""
    s = html.unescape(html.unescape(s))
    s = s.replace("[?]", "₹")                       # lost rupee sign in some FAQ answers
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", s)   # [text](url) -> text
    s = re.sub(r"^\s*>\s?", "", s, flags=re.M)      # blockquote markers
    s = re.sub(r"\*+", "", s)                       # bold/italic markers
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def labels(items):
    return [i.get("label", "") for i in (items or []) if isinstance(i, dict) and i.get("label")]

def parse(rec):
    """Return a flat clean dict from one raw record, or None if unusable."""
    det = (rec.get("scheme_detail") or {}).get("en") or {}
    basic = det.get("basicDetails") or {}
    content = det.get("schemeContent") or {}
    fields = (rec.get("listing_record") or {}).get("fields") or {}
    name = clean(basic.get("schemeName") or fields.get("schemeName") or "")
    if not name:
        return None
    slug = (rec.get("scheme_detail") or {}).get("slug") or rec.get("slug") or fields.get("slug")
    level = (basic.get("level") or {}).get("label") or fields.get("level") or ""
    state = (basic.get("state") or {}).get("label") or ", ".join(fields.get("beneficiaryState") or [])
    proc = []
    for p in det.get("applicationProcess") or []:
        txt = clean(p.get("process_md"))
        if txt:
            proc.append({"mode": p.get("mode", ""), "text": txt})
    faqs = []
    for q in ((rec.get("faqs") or {}).get("en") or {}).get("faqs") or []:
        qq, aa = clean(q.get("question")), clean(q.get("answer_md"))
        if qq and aa:
            faqs.append({"question": qq, "answer": aa})
    return {
        "slug": slug, "name": name, "short_title": basic.get("schemeShortTitle", ""),
        "level": level, "state": state,
        "department": (basic.get("nodalDepartmentName") or {}).get("label") or fields.get("nodalDepartmentName", ""),
        "categories": labels(basic.get("schemeCategory")) or fields.get("schemeCategory") or [],
        "beneficiaries": labels(basic.get("targetBeneficiaries")),
        "tags": basic.get("tags") or [],
        "brief": clean(content.get("briefDescription") or fields.get("briefDescription")),
        "details": clean(content.get("detailedDescription_md")),
        "benefits": clean(content.get("benefits_md")),
        "eligibility": clean((det.get("eligibilityCriteria") or {}).get("eligibilityDescription_md")),
        "exclusions": clean(content.get("exclusions_md")),
        "process": proc,
        "documents": clean(((rec.get("documents") or {}).get("en") or {}).get("documentsRequired_md")),
        "faqs": faqs,
        "references": [{"title": r.get("title", ""), "url": r.get("url", "")} for r in content.get("references") or []],
        "url": rec.get("source_url") or (f"https://www.myscheme.gov.in/schemes/{slug}" if slug else ""),
        "scraped_at": rec.get("scraped_at", ""),
        "status": rec.get("collection_status", ""),
    }

def split_text(text, limit=MAX_CHARS):
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > limit:
            parts.append(cur.strip()); cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts

def chunks_for(s, faq_in_index=True):
    where = s["state"] if s["level"].lower().startswith("state") and s["state"] else s["level"]
    short = (s.get("short_title") or "").strip()
    label = s["name"] + (f" ({short})" if short and short.lower() not in s["name"].lower() else "")
    header = f'Scheme: {label} ({where})'
    sections = [
        ("Overview", "\n".join(x for x in [s["brief"], s["details"]] if x)),
        ("Benefits", s["benefits"]),
        ("Eligibility", s["eligibility"]),
        ("Exclusions", s["exclusions"]),
        ("Documents required", s["documents"]),
    ] + [(f'How to apply ({p["mode"]})' if p["mode"] else "How to apply", p["text"]) for p in s["process"]]
    out = []
    for sec, text in sections:
        for i, part in enumerate(split_text(text) if text else []):
            out.append((sec, f"{header} | Section: {sec}\n{part}"))
    if faq_in_index:
        for f in s["faqs"]:
            out.append(("FAQ", f'{header} | Section: FAQ\nQ: {f["question"]}\nA: {f["answer"]}'))
    return [{
        "id": f'{s["slug"]}::{n}', "text": t, "scheme": s["name"], "slug": s["slug"], "section": sec,
        "url": s["url"], "references": s["references"], "level": s["level"], "state": s["state"],
        "categories": s["categories"], "retrieved": s["scraped_at"][:10],
    } for n, (sec, t) in enumerate(out)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--stats", action="store_true", help="print dataset overview and exit")
    ap.add_argument("--slugs-file", help="text file, one slug per line")
    ap.add_argument("--categories", nargs="*", help="keep schemes whose category contains any of these words")
    ap.add_argument("--level", choices=["Central", "State"], help="keep only central or state schemes")
    ap.add_argument("--state", help="keep only this state/UT (substring match)")
    ap.add_argument("--limit", type=int, help="random sample of N schemes after filtering")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=400, help="drop schemes with very little text")
    ap.add_argument("--exclude-faq-from-index", action="store_true",
                    help="keep FAQs out of chunks.json so they can be used as unseen evaluation questions")
    a = ap.parse_args()

    seen, kept, bad, incomplete, thin = set(), [], 0, 0, 0
    with open(a.jsonl, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1; continue
            s = parse(rec)
            if not s or not s["slug"] or s["slug"] in seen:
                continue
            seen.add(s["slug"])
            if s["status"] and s["status"] != "complete":
                incomplete += 1; continue
            body = " ".join([s["brief"], s["details"], s["benefits"], s["eligibility"], s["documents"]])
            if len(body) < a.min_chars:
                thin += 1; continue
            kept.append(s)

    print(f"usable schemes: {len(kept)} | bad lines: {bad} | incomplete: {incomplete} | too thin: {thin}")
    if a.stats:
        from collections import Counter
        print("level:", Counter(s["level"] for s in kept).most_common())
        cats = Counter(c for s in kept for c in s["categories"])
        print("top categories:", cats.most_common(12))
        print("top states:", Counter(s["state"] for s in kept).most_common(8))
        return

    if a.slugs_file:
        want = {x.strip() for x in Path(a.slugs_file).read_text().splitlines() if x.strip()}
        kept = [s for s in kept if s["slug"] in want]
    if a.categories:
        kept = [s for s in kept if any(w.lower() in c.lower() for c in s["categories"] for w in a.categories)]
    if a.level:
        kept = [s for s in kept if s["level"].lower().startswith(a.level.lower())]
    if a.state:
        kept = [s for s in kept if a.state.lower() in s["state"].lower()]
    if a.limit and len(kept) > a.limit:
        random.Random(a.seed).shuffle(kept); kept = kept[:a.limit]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    chunks = [c for s in kept for c in chunks_for(s, faq_in_index=not a.exclude_faq_from_index)]
    evalc = [{"slug": s["slug"], "scheme": s["name"], "question": f["question"], "reference_answer": f["answer"], "url": s["url"]}
             for s in kept for f in s["faqs"]]
    json.dump(chunks, open(out / "chunks.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    json.dump(kept, open(out / "schemes_clean.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    json.dump(evalc, open(out / "eval_candidates.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"selected {len(kept)} schemes -> {len(chunks)} chunks, {len(evalc)} FAQ eval candidates in {out}/")

if __name__ == "__main__":
    main()
