"""
Answer generation pipeline: retrieve -> threshold check -> prompt -> generate -> format citations.
"""
from dataclasses import dataclass
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

import yaml

from retrieve import retrieve
from llm import LLMProvider, get_default_provider, create_provider


@dataclass
class AnswerResult:
    """Result of answer generation."""
    answer: str
    citations: List[Dict[str, Any]]
    best_score: float
    retrieved_chunks: int
    provider: str
    language: str
    abstained: bool = False
    abstain_reason: Optional[str] = None


class AnswerGenerator:
    """Main pipeline for generating answers with citations."""
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        provider_name: Optional[str] = None,
    ):
        self.config_path = config_path
        
        if provider_name:
            self.llm = create_provider(provider_name, config_path)
            self.provider_name = provider_name
        else:
            self.llm = get_default_provider(config_path)
            with open(config_path, "r") as f:
                import yaml
                config = yaml.safe_load(f)
            self.provider_name = config.get("default_provider", "local")
        
        # Load threshold from config
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.score_threshold = config["retrieval"]["score_threshold"]
        self.config = config
        self.answer_config = config.get("answer", {})
        self._indexed_chunks: Optional[List[Dict[str, Any]]] = None
    
    def generate(
        self,
        question: str,
        language: str = "English",
        detail: str = "brief",
    ) -> AnswerResult:
        """
        Generate an answer for a question.
        
        Args:
            question: User question
            language: "English" or "Hindi"
        
        Returns:
            AnswerResult with answer, citations, and metadata
        """
        if detail not in ("brief", "detailed"):
            raise ValueError("detail must be 'brief' or 'detailed'")
        mode = self.answer_config.get(detail, {})

        # Retrieve relevant chunks
        chunks, best_score = retrieve(question, config_path=self.config_path)
        
        # Check if we have sufficient retrieval confidence
        if best_score < self.score_threshold or not chunks:
            return AnswerResult(
                answer="not found" if language == "English" else "नहीं मिला",
                citations=[],
                best_score=best_score,
                retrieved_chunks=len(chunks),
                provider=self.provider_name,
                language=language,
                abstained=True,
                abstain_reason="low_score",
            )
        
        focus_slug = chunks[0].get("slug")
        focus_scheme = chunks[0].get("scheme")
        focus_gap = self.answer_config.get("focus_gap", 0.02)
        top_score = chunks[0].get("score", best_score)
        other_scheme_scores = [
            chunk.get("score", 0.0)
            for chunk in chunks
            if chunk.get("slug") != focus_slug
        ]
        ambiguous = any(top_score - score <= focus_gap for score in other_scheme_scores)

        if detail == "detailed" and mode.get("expand_scheme_sections", False):
            chunks = self._expand_scheme_chunks(chunks, mode)
            if not ambiguous and focus_slug:
                chunks = [chunk for chunk in chunks if chunk.get("slug") == focus_slug]

        max_context_chars = mode.get(
            "max_context_chars",
            self.config["retrieval"].get("max_context_chars", 4000),
        )
        context, context_chunks = self._format_context(chunks, max_context_chars)
        response_instructions = self._response_instructions(detail, mode.get("max_words", 100))
        
        # Generate answer with LLM
        answer = self.llm.generate_with_citations(
            question,
            context,
            language,
            response_instructions=response_instructions,
            num_predict=mode.get("num_predict"),
        )
        
        # Check if LLM also says "not found"
        abstained = answer.strip().lower() in ["not found", "नहीं मिला"]
        
        abstain_reason = "model_not_found" if abstained else None
        cited_numbers = {
            int(number)
            for number in re.findall(r"\[(\d+)\]", answer)
            if 1 <= int(number) <= len(context_chunks)
        }
        citations = [] if abstained else [
            {
                "index": i + 1,
                "slug": chunk.get("slug"),
                "scheme": chunk["scheme"],
                "section": chunk["section"],
                "level": chunk.get("level"),
                "state": chunk.get("state"),
                "url": chunk["url"],
                "retrieved": chunk["retrieved"],
                "score": chunk["score"],
            }
            for i, chunk in enumerate(context_chunks)
            if i + 1 in cited_numbers
        ]
        if detail == "detailed" and not ambiguous and not abstained and focus_slug:
            other_cited_schemes = list(dict.fromkeys(
                cite["scheme"] for cite in citations if cite.get("slug") != focus_slug
            ))
            for scheme_name in other_cited_schemes:
                answer += (
                    f"\n\nNote: part of this answer cites another scheme ({scheme_name}). "
                    "Verify on the official portal."
                )
        
        return AnswerResult(
            answer=answer,
            citations=citations,
            best_score=best_score,
            retrieved_chunks=len(chunks),
            provider=self.provider_name,
            language=language,
            abstained=abstained,
            abstain_reason=abstain_reason,
        )
    
    def _format_context(
        self, chunks: List[Dict[str, Any]], max_chars: int = 4000
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Format chunks and return only passages included in the prompt."""
        chunks = self._add_chunk_locations(chunks)
        context_parts = []
        included_chunks = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks):
            location = chunk.get("level") or chunk.get("state") or "Unknown"
            citation = (
                f"[{i+1}] {chunk['scheme']} ({location}), {chunk['section']}: "
                f"URL: {chunk['url']} | Retrieved: {chunk['retrieved']}"
            )
            chunk_text = f"{citation}\n{chunk['text']}"
            
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            included_chunks.append(chunk)
            total_chars += len(chunk_text)
        
        return "\n\n---\n\n".join(context_parts), included_chunks

    def _load_index_chunks(self) -> List[Dict[str, Any]]:
        """Load and cache full chunk metadata from the index file."""
        if self._indexed_chunks is None:
            chunks_path = Path(self.config["index"]["chunks_path"])
            with chunks_path.open("r", encoding="utf-8") as stream:
                self._indexed_chunks = json.load(stream)
        return self._indexed_chunks

    def _add_chunk_locations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill level and state metadata omitted from retrieval results."""
        locations = {}
        for indexed in self._load_index_chunks():
            slug = indexed.get("slug")
            if slug and slug not in locations:
                locations[slug] = (indexed.get("level"), indexed.get("state"))
        enriched = []
        for chunk in chunks:
            item = dict(chunk)
            level, state = locations.get(item.get("slug"), (None, None))
            item.setdefault("level", level)
            item.setdefault("state", state)
            enriched.append(item)
        return enriched

    def _expand_scheme_chunks(
        self,
        chunks: List[Dict[str, Any]],
        mode: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Append ordered sections for the highest-ranked schemes."""
        top_count = mode.get("expand_top_schemes", 2)
        slugs = list(dict.fromkeys(chunk.get("slug") for chunk in chunks if chunk.get("slug")))[:top_count]
        if not slugs:
            return chunks

        indexed_chunks = self._load_index_chunks()

        section_order = (
            ("overview", ("overview", "brief description", "scheme details")),
            ("benefits", ("benefit",)),
            ("eligibility", ("eligibility",)),
            ("documents required", ("document",)),
            ("how to apply", ("how to apply", "application", "process")),
        )
        existing = {
            (chunk.get("slug"), chunk.get("section"), chunk.get("text"))
            for chunk in chunks
        }
        expanded = list(chunks)
        for slug in slugs:
            scheme_chunks = [chunk for chunk in indexed_chunks if chunk.get("slug") == slug]
            for _, keywords in section_order:
                for chunk in scheme_chunks:
                    section = str(chunk.get("section", "")).lower()
                    key = (chunk.get("slug"), chunk.get("section"), chunk.get("text"))
                    if key not in existing and any(word in section for word in keywords):
                        item = dict(chunk)
                        item.setdefault("score", 0.0)
                        expanded.append(item)
                        existing.add(key)
        return expanded

    @staticmethod
    def _response_instructions(detail: str, max_words: int) -> str:
        """Build mode-specific answer instructions while preserving project rules."""
        shared = (
            f"Use no more than {max_words} words. Use only the provided passages, cite passage numbers, "
            "answer in the question's language, say not found when the passages do not contain the answer, "
            "and do not decide eligibility for the user. If the passages do not state a detail, do not mention it. "
            "Never use outside knowledge. Each passage names its scheme. Answer only about the scheme the question asks about. "
            "Do not use a passage about a different scheme, even if it mentions the asked scheme. "
            "Do not add statements about what a scheme does not provide. "
            "If the question is ambiguous between schemes, name each scheme next to its facts."
        )
        if detail == "detailed":
            return (
                f"{shared} Begin with a one-sentence direct answer, then use short bullet points covering every "
                "relevant amount, condition, exception, document, and step found in the passages."
            )
        return shared
    
    def format_answer(self, result: AnswerResult) -> str:
        """Format an answer with only cited sources and the correct abstention reason."""
        output = result.answer

        if result.citations and not result.abstained:
            output += "\n\n**Sources:**"
            for cite in result.citations:
                location = cite.get("level") or cite.get("state")
                location_text = f" | {location}" if location else ""
                output += (
                    f"\n[{cite['index']}] {cite['scheme']} ? {cite['section']}"
                    f"{location_text} | {cite['url']} | retrieved {cite['retrieved']}"
                )
        elif not result.abstained:
            output += "\n\nNo source passage was cited; verify on the official portal."

        if result.abstain_reason == "low_score":
            output += (
                f"\n\nThe best match score was {result.best_score:.3f}, "
                "below the confidence threshold."
            )
        elif result.abstain_reason == "model_not_found":
            output += (
                "\n\nThe assistant could not find this in the collected scheme documents "
                f"(best match score {result.best_score:.3f}). Please check the official portal."
            )

        return output


def main():
    """Test the answer generator."""
    generator = AnswerGenerator()
    
    test_questions = [
        "What are the benefits of PM-KISAN scheme?",
        "How to apply for Ayushman Bharat?",
        "What is the eligibility for Sukanya Samriddhi Yojana?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        result = generator.generate(question)
        formatted = generator.format_answer(result)
        print(formatted)
        print(f"\n[Provider: {result.provider}, Score: {result.best_score:.3f}, Chunks: {result.retrieved_chunks}, Abstained: {result.abstained}]")


if __name__ == "__main__":
    main()
