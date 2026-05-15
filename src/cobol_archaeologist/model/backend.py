"""Pluggable LLM backends.

Backends:
- ``EchoBackend``: deterministic, no network. Used for tests and offline demos.
- ``HFBackend``: local HuggingFace ``transformers`` model.
- ``OpenAIBackend``: OpenAI-compatible HTTP API (works with OpenAI, vLLM, LM Studio).
- ``OllamaBackend``: local Ollama daemon.
- ``LlamaCppBackend``: local GGUF model via llama-cpp-python (CPU/GPU).

All backends share the same ``generate(prompt) -> str`` contract.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class EchoBackend:
    """Deterministic backend that produces a plausible Business Intent Card.

    It looks at the prompt for variables/conditions and emits a JSON object.
    Useful for local development without GPUs and for unit tests.
    """

    def generate(self, prompt: str) -> str:
        block_match = re.search(r"COBOL CODE:\s*(.+?)\nSTATIC CONTEXT:", prompt, re.S)
        ctx_match = re.search(r"STATIC CONTEXT:\s*(\{.*?\})\nRETRIEVED REGULATIONS", prompt, re.S)
        block = block_match.group(1).strip() if block_match else ""
        ctx = {}
        if ctx_match:
            try:
                ctx = json.loads(ctx_match.group(1))
            except json.JSONDecodeError:
                ctx = {}

        vars_read = ctx.get("vars_read", []) or []
        vars_written = ctx.get("vars_written", []) or []
        conditions = ctx.get("conditions", []) or []

        evidence: list[str] = []
        for v in vars_read[:3]:
            evidence.append(f"Reads {v}.")
        for v in vars_written[:3]:
            evidence.append(f"Writes {v}.")
        for c in conditions[:2]:
            evidence.append(f"Branches on condition: {c}.")

        what = "Performs business logic over the listed COBOL variables."
        why = "Likely enforces a domain or compliance rule based on referenced fields."
        if any("BALANCE" in v for v in vars_read + vars_written):
            what = "Validates whether a transaction can proceed against the available balance."
            why = "Prevents overdrafts and supports transaction approval policy."

        card = {
            "what": what,
            "why": why,
            "code_evidence": evidence or [block[:120]],
            "regulation_link": None,
            "regulation_sources": [],
            "confidence": {
                "level": "Medium",
                "justification": "Heuristic offline backend; no model reasoning applied.",
            },
        }
        return json.dumps(card)


class HFBackend:
    """HuggingFace transformers backend (lazy import)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_new_tokens: int = 512,
        device: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        import torch  # type: ignore

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


class OpenAIBackend:
    """OpenAI-compatible HTTP API backend (works with OpenAI, vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        import requests  # type: ignore

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You output strict JSON Business Intent Cards."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class OllamaBackend:
    """Local Ollama daemon backend."""

    def __init__(self, model: str = "qwen2.5-coder:1.5b", host: str | None = None) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(self, prompt: str) -> str:
        import requests  # type: ignore

        r = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("response", "")


class LlamaCppBackend:
    """Local GGUF model backend via llama-cpp-python.

    Loads the model once and keeps it in memory. Supports both chat and
    raw-completion modes.  Pass ``n_gpu_layers=-1`` to offload all layers to GPU
    when CUDA is available.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        from llama_cpp import Llama  # type: ignore

        if model_path is None:
            model_path = os.environ.get(
                "LLAMA_MODEL_PATH",
                str(
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../../outputs/cobol-archaeologist.f16.gguf",
                    )
                ),
            )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        result = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "<|im_end|>"],
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """OpenAI-style chat completion."""
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "<|im_end|>"],
        )
        return result["choices"][0]["message"]["content"].strip()


def get_backend(name: str = "echo", **kwargs) -> LLMBackend:
    name = name.lower()
    if name == "echo":
        return EchoBackend()
    if name == "hf":
        return HFBackend(**kwargs)
    if name in {"openai", "openai-compat"}:
        return OpenAIBackend(**kwargs)
    if name == "ollama":
        return OllamaBackend(**kwargs)
    if name in {"llama", "llama-cpp", "gguf"}:
        return LlamaCppBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")
