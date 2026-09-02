"""Base class for all caption rewriter agents."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from abc import ABC, abstractmethod
from typing import Any

from ..models import CaptionState


class BaseAgent(ABC):
    """All agents share the same LLM and state object."""

    def __init__(self, llm_model: Any, state: CaptionState, *, style_keywords: str = "", lyrics: str = "") -> None:
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
        """Call the external LLM model and log the raw response for debugging.

        Supports three interfaces:
        1. ComfyUI CLIP-wrapped model (e.g. loaded via ComfyUI's own
           CLIPLoader, per comfy_extras/nodes_textgen.py's TextGenerate node):
           calls clip.tokenize()/generate()/decode() directly.
        2. Hugging Face style: llm_model has 'model', 'tokenizer', 'device' keys
        3. Simple callable: llm_model['generate'](prompt) returns a string

        Falls back to a simple split on '###' headers if no LLM is provided.
        """
        response, generated = self._generate(prompt)
        if generated:
            logging.info(
                "[music_caption:%s] LLM response:\n%s", type(self).__name__, response
            )
        return response

    def _generate(self, prompt: str) -> tuple[str, bool]:
        """Returns (response, generated) -- generated is False for the
        no-LLM prompt-echo fallback, so _call_llm knows not to log it.
        """
        if self.llm_model is None:
            return prompt, False

        if (
            hasattr(self.llm_model, "tokenize")
            and hasattr(self.llm_model, "generate")
            and hasattr(self.llm_model, "decode")
        ):
            # skip_template=False applies the model's chat-turn wrapping
            # (e.g. Gemma's <start_of_turn>user...<end_of_turn>), which most
            # tokenizers default to skipping (comfy/text_encoders/lt.py's
            # Gemma3_Tokenizer defaults skip_template=True). Without it the
            # model gets a raw, un-turned prompt with no stop cue and rambles
            # until it hits max_length instead of finishing naturally.
            tokens = self.llm_model.tokenize(prompt, skip_template=False)
            generated_ids = self.llm_model.generate(
                tokens,
                do_sample=True,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                # CLIP.generate defaults seed=None, which crashes
                # torch.Generator.manual_seed() whenever do_sample=True
                # (comfy/text_encoders/llama.py's BaseGenerate.generate) --
                # an explicit seed is required, not optional.
                seed=random.randint(0, 0xFFFFFFFFFFFFFFFF),
            )
            return self.llm_model.decode(generated_ids), True

        if not self.llm_model or "generate" not in self.llm_model:
            # No LLM provided — return prompt as-is for testing
            return prompt, False

        gen = self.llm_model["generate"]
        if callable(gen):
            return gen(prompt), True

        # HF-style generation
        model = self.llm_model.get("model")
        tokenizer = self.llm_model.get("tokenizer")
        device = self.llm_model.get("device", "cpu")

        if model is None or tokenizer is None:
            return prompt, False

        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)

        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True), True

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
