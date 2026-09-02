"""Stage 1 — BriefAgent: Build Music Brief from style keywords and lyrics."""

from __future__ import annotations

import re

from .base import BaseAgent
from ..models import CaptionState, Confidence


class BriefAgent(BaseAgent):
    """Extracts a structured Music Brief from the user's inputs."""

    SYSTEM_PROMPT = (
        "You are a music brief extractor for MiniMax Music 3. "
        "Given style keywords and optional lyrics body text, extract a structured musical description. "
        "Do not invent exact BPM, key, or vocal gender without evidence. "
        "Do not use markdown formatting (no bullets, no bold/asterisks, no headers) "
        "and do not add any prose before or after the list of lines."
    )

    FIELDS = [
        "macro_genre",
        "subgenres",
        "cultural_style",
        "mood_arc",
        "tempo",
        "meter",
        "groove",
        "vocal_presence",
        "vocal_gender",
        "vocal_register",
        "vocal_timbre",
        "vocal_delivery",
        "core_instruments",
        "production_texture",
        "spatial_character",
        "explicit_exclusions",
    ]

    def build_prompt(self) -> str:
        brief = self.state.music_brief
        style = brief.macro_genre or ""
        subgenres = ", ".join(brief.subgenres) if brief.subgenres else ""
        mood = brief.mood_arc or ""
        instruments = ", ".join(brief.core_instruments) if brief.core_instruments else ""

        parts = []
        if style:
            parts.append(f"Macro genre: {style}")
        if subgenres:
            parts.append(f"Subgenres: {subgenres}")
        if mood:
            parts.append(f"Mood: {mood}")
        if instruments:
            parts.append(f"Instruments: {instruments}")

        lyrics_body = ""
        if self._lyrics:
            # Extract body text from lyrics (remove bracketed tags)
            lyrics_body = re.sub(r"\[.*?\]", "", self._lyrics).strip()

        prompt = self.SYSTEM_PROMPT + "\n\n"
        prompt += f"Style keywords: {self._style_keywords}\n"
        if lyrics_body:
            prompt += f"Lyrics body: {lyrics_body}\n"
        if parts:
            prompt += f"Already identified: {'; '.join(parts)}\n"

        field_list = "\n".join(f"- {name}" for name in self.FIELDS)
        prompt += (
            "\nRespond with exactly one line per field below, using the exact "
            "field name shown (do not rename, merge, or split fields), in this "
            f"exact format:\n{field_list}\n\n"
            "Format: <field_name>: <value> (<CONFIDENCE>)\n"
            "<CONFIDENCE> must be exactly one of: EXPLICIT, TAGGED, INFERRED, UNSPECIFIED.\n"
            "For subgenres, core_instruments, and explicit_exclusions, use a "
            "comma-separated list as <value>.\n"
            "Example:\n"
            "macro_genre: deep house (EXPLICIT)\n"
            "vocal_gender: female (INFERRED)\n"
            "core_instruments: analog synth, deep bass (TAGGED)\n"
        )
        return self._sanitize(prompt)

    def parse_response(self, response: str) -> CaptionState:
        brief = self.state.music_brief

        for field in self.FIELDS:
            line = self._extract_field(response, field)
            # Also extract confidence if present
            conf_match = re.search(r"\((EXPLICIT|TAGGED|INFERRED|UNSPECIFIED)\)", line, re.IGNORECASE)
            confidence = Confidence.INFERRED  # default
            if conf_match:
                try:
                    confidence = Confidence(conf_match.group(1).upper())
                except ValueError:
                    confidence = Confidence.INFERRED
                line = re.sub(r"\s*\([^)]+\)", "", line)

            if field in ("subgenres", "core_instruments", "explicit_exclusions"):
                items = [i.strip() for i in line.split(",") if i.strip()]
                setattr(brief, field, items)
            else:
                setattr(brief, field, line or "unspecified")

            brief.set_confidence(field, confidence)

        # Extract section structure from explicit sections list
        if self.state.explicit_sections:
            brief.section_structure = list(self.state.explicit_sections)

        return self.state

    async def run(self) -> CaptionState:
        """Execute the brief extraction LLM call."""
        prompt = self.build_prompt()
        response = self._call_llm(prompt)
        return self.parse_response(response)
