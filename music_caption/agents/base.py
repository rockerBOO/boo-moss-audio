"""Base class for all caption rewriter agents."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod

from ..models import CaptionState


class BaseAgent(ABC):
    """All agents share the same LLM and state object."""

    def __init__(self, llm_model: dict, state: CaptionState, *, style_keywords: str = "", lyrics: str = "") -> None:
        self.llm_model = llm_model
        self.state = state
        self._style_keywords = style_keywords
        self._lyrics = lyrics

    @abstractmethod
    async def run(self) -> CaptionState:
        """Execute the agent's logic and return the updated state."""

    @abstractmethod
    def build_prompt(self) -> str:
        """Build the prompt for this agent's LLM call."""

    @abstractmethod
    def parse_response(self, response: str) -> CaptionState:
        """Parse the LLM response and update state."""

    def _call_llm(self, prompt: str) -> str:
        """Call the external LLM model.

        Supports two interfaces:
        1. Hugging Face style: llm_model has 'model', 'tokenizer', 'device' keys
        2. Simple callable: llm_model['generate'](prompt) returns a string

        Falls back to a simple split on '###' headers if no LLM is provided.
        """
        if not self.llm_model or "generate" not in self.llm_model:
            # No LLM provided — return prompt as-is for testing
            return prompt

        gen = self.llm_model["generate"]
        if callable(gen):
            return gen(prompt)

        # HF-style generation
        model = self.llm_model.get("model")
        tokenizer = self.llm_model.get("tokenizer")
        device = self.llm_model.get("device", "cpu")

        if model is None or tokenizer is None:
            return prompt

        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)

        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    def _extract_section(self, text: str, header: str) -> str:
        """Extract content under a ### header from LLM output."""
        pattern = rf"###\s*{re.escape(header)}\s*\n(.*?)(?=###|\Z)", re.DOTALL
        match = re.search(pattern[0], text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a 'Field: value' line from LLM output."""
        pattern = rf"{re.escape(field_name)}\s*[:=]\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_field_list(self, text: str, field_name: str) -> list[str]:
        """Extract a comma-separated or bulleted list for a field."""
        value = self._extract_field(text, field_name)
        if not value:
            return []
        # Split on commas, semicolons, or newlines
        items = re.split(r"[,;]", value)
        return [item.strip() for item in items if item.strip()]

    def _sanitize(self, text: str) -> str:
        """Remove common LLM formatting artifacts."""
        # Remove markdown code fences
        text = re.sub(r"```[\s\S]*?\n", "", text)
        # Remove extra blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
