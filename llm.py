"""
LLM provider abstraction for native local Ollama and cloud Groq.
"""
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: str
    model: str
    temperature: float = 0.1
    base_url: Optional[str] = None
    max_tokens: int = 1024
    reasoning_effort: Optional[str] = None
    api_key: Optional[str] = None
    num_predict: int = 300
    num_ctx: int = 4096
    num_gpu: int = 0
    num_thread: int = 4
    think: Optional[bool] = False


def _clean_ollama_text(text: str) -> str:
    """Keep the final answer and remove tagged reasoning."""
    if "</think>" in text:
        text = text.rsplit("</think>", maxsplit=1)[-1]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _looks_like_reasoning(text: str) -> bool:
    """Recognize common reasoning prefixes in a partial response."""
    return bool(
        re.match(
            r"^(?:(?:okay|hmm)[,.! ]*)?(?:"
            r"let me (?:think|try to|figure out|analy[sz]e)|"
            r"let's (?:think|reason|analy[sz]e)|"
            r"the user is asking|the question asks|"
            r"i need to (?:think|check|consider)|"
            r"first[, :]?\s+(?:i|we)\b|"
            r"आइए सोचें|मुझे (?:सोचना|देखना|जांचना)|पहले (?:मुझे|मैं)"
            r")",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def _process_ollama_response(text: str, done_reason: Optional[str]) -> str:
    """Clean an Ollama response or reject incomplete reasoning."""
    answer = _clean_ollama_text(text)
    if done_reason == "length" and (not answer or _looks_like_reasoning(answer)):
        raise RuntimeError("Ollama ran out of tokens while reasoning; increase num_predict.")
    if not answer:
        raise RuntimeError("Ollama returned an empty answer after cleaning think output.")
    return answer


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        pass
    
    def generate_with_citations(
        self,
        question: str,
        context: str,
        language: str = "English",
        response_instructions: str = "",
        num_predict: Optional[int] = None,
    ) -> str:
        """Generate an answer with citations from retrieved context."""
        system_prompt = self._get_system_prompt(language)
        prompt = f"""Answer the question using ONLY the provided context. 
If the answer is not in the context, respond with "not found" in the requested language.

Context:
{context}

Question: {question}

Answer in {language} with citations like [1], [2] referencing the context sources:"""
        if response_instructions:
            prompt = f"{response_instructions}\n\n{prompt}"
        return self.generate(prompt, system_prompt, num_predict=num_predict)

    def _get_system_prompt(self, language: str) -> str:
        """Get the system prompt for a language."""
        if language == "Hindi":
            return """आप भारत सरकार की योजनाओं के बारे में प्रश्नों का उत्तर देने वाले एक सहायक हैं।
केवल प्रदान किए गए आधिकारिक पाठ का उपयोग करके उत्तर दें।
यदि जानकारी संदर्भ में नहीं है, तो "नहीं मिला" लिखें।
उत्तर हिंदी में दें और संदर्भ उद्धरण [1], [2] आदि का उपयोग करें।"""
        return """You are an assistant that answers questions about Indian government schemes.
Use ONLY the provided official text to answer.
If the information is not in the context, respond with "not found".
Answer in English and use citation references [1], [2], etc."""


class OpenAICompatibleProvider(LLMProvider):
    """Base class for OpenAI-compatible providers such as Groq."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.base_url is None:
            raise ValueError("base_url must be configured for this provider")
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "ollama",  # Ollama doesn't need real key
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """Generate a response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_options = {}
        if self.config.reasoning_effort:
            request_options.update(
                reasoning_effort=self.config.reasoning_effort,
                include_reasoning=False,
            )

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **request_options,
        )
        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        if choice.finish_reason == "length" and not answer:
            raise RuntimeError("Groq model ran out of tokens while reasoning; increase max_tokens.")
        return answer


class OllamaProvider(LLMProvider):
    """Ollama local provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """Generate a response with the native Ollama client."""
        import ollama

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        try:
            chat_options = {
                "model": self.config.model,
                "messages": messages,
                "options": {
                    "num_gpu": self.config.num_gpu,
                    "num_ctx": self.config.num_ctx,
                    "num_predict": num_predict if num_predict is not None else self.config.num_predict,
                    "num_thread": self.config.num_thread,
                    "temperature": self.config.temperature,
                },
            }
            if self.config.think is not None:
                chat_options["think"] = self.config.think
            response = ollama.chat(**chat_options)
            return _process_ollama_response(
                response.message.content or "",
                response.done_reason,
            )
        finally:
            elapsed = time.perf_counter() - started
            if os.environ.get("LOG_LEVEL", "").lower() == "debug":
                print(f"Ollama call completed in {elapsed:.2f} seconds")


class GroqProvider(OpenAICompatibleProvider):
    """Groq cloud provider."""
    
    def __init__(self, config: LLMConfig):
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to .env.")
        config.api_key = api_key
        super().__init__(config)


def load_llm_config(
    config_path: str | Path = "config.yaml",
    provider_name: Optional[str] = None,
) -> LLMConfig:
    """Load LLM configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if provider_name is None:
        provider_name = config.get("default_provider", "local")
    
    provider_config = config["llm"][provider_name]
    
    return LLMConfig(
        provider=provider_config["provider"],
        model=provider_config["model"],
        base_url=provider_config.get("base_url"),
        temperature=provider_config.get("temperature", 0.1),
        max_tokens=provider_config.get("max_tokens", 1024),
        reasoning_effort=provider_config.get("reasoning_effort"),
        num_predict=provider_config.get("num_predict", 300),
        num_ctx=provider_config.get("num_ctx", 4096),
        num_gpu=provider_config.get("num_gpu", 0),
        num_thread=provider_config.get("num_thread", 4),
        think=provider_config.get("think", False),
    )


def create_provider(provider_name: str, config_path: str | Path = "config.yaml") -> LLMProvider:
    """Factory function to create LLM provider."""
    llm_config = load_llm_config(config_path, provider_name)
    
    if llm_config.provider == "ollama":
        return OllamaProvider(llm_config)
    elif llm_config.provider == "groq":
        return GroqProvider(llm_config)
    else:
        raise ValueError(f"Unknown provider: {llm_config.provider}")


def get_default_provider(config_path: str | Path = "config.yaml") -> LLMProvider:
    """Get the default provider from config."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    default = config.get("default_provider", "local")
    return create_provider(default, config_path)


def _run_selftest() -> None:
    """Run mocked Ollama response-cleaning checks."""
    from types import SimpleNamespace
    from unittest.mock import patch

    provider = OllamaProvider(LLMConfig(provider="ollama", model="qwen3:4b", think=None))
    cases = (
        ("partial thoughts</think>Final answer", "stop", "Final answer"),
        ("<think>hidden\nreasoning</think>Final answer", "stop", "Final answer"),
        ("  Direct answer  ", "stop", "Direct answer"),
        ("Plain answer without think tags.", "stop", "Plain answer without think tags."),
    )
    for content, done_reason, expected in cases:
        response = SimpleNamespace(
            message=SimpleNamespace(content=content),
            done_reason=done_reason,
        )
        with patch("ollama.chat", return_value=response) as mock_chat:
            assert provider.generate("test") == expected
            assert "think" not in mock_chat.call_args.kwargs

    empty_response = SimpleNamespace(
        message=SimpleNamespace(content="<think>hidden\nreasoning</think>"),
        done_reason="stop",
    )
    with patch("ollama.chat", return_value=empty_response):
        try:
            provider.generate("test")
        except RuntimeError as error:
            assert "empty answer" in str(error)
        else:
            raise AssertionError("Expected an error for an empty cleaned answer")

    empty_length_response = SimpleNamespace(
        message=SimpleNamespace(content="<think>hidden\nreasoning</think>"),
        done_reason="length",
    )
    with patch("ollama.chat", return_value=empty_length_response):
        try:
            provider.generate("test")
        except RuntimeError as error:
            assert "ran out of tokens while reasoning" in str(error)
        else:
            raise AssertionError("Expected an error for an empty length-limited response")

    reasoning_response = SimpleNamespace(
        message=SimpleNamespace(content="Okay, let me try to figure out the answer."),
        done_reason="length",
    )
    with patch("ollama.chat", return_value=reasoning_response):
        try:
            provider.generate("test")
        except RuntimeError as error:
            assert "ran out of tokens while reasoning" in str(error)
        else:
            raise AssertionError("Expected an error for truncated reasoning")

    print("Ollama response self-test passed.")


def _run_examples() -> None:
    """Run provider smoke tests."""
    # Test with local provider
    try:
        provider = create_provider("local")
        response = provider.generate("Hello, how are you?")
        print(f"Local response: {response}")
    except Exception as e:
        print(f"Local provider error: {e}")
    
    # Test with Groq provider (if API key set)
    if os.environ.get("GROQ_API_KEY"):
        try:
            provider = create_provider("groq")
            response = provider.generate("Hello, how are you?")
            print(f"Groq response: {response}")
        except Exception as e:
            print(f"Groq provider error: {e}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _run_selftest()
    else:
        _run_examples()
