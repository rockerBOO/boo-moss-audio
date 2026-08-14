"""Stage 2 — ConstraintsAgent: Resolve constraints with precedence rules."""

from __future__ import annotations

import re

from .base import BaseAgent
from ..models import CaptionState


class ConstraintsAgent(BaseAgent):
    """Resolves explicit constraints and section-local directives."""

    SYSTEM_PROMPT = (
        "You are a constraint resolver for MiniMax Music 3. "
        "Given a Music Brief and optional section tags from lyrics, apply this precedence: "
        "1) Explicit user requirements and exclusions. "
        "2) Section-local directives from lyric tags. "
        "3) Strong implications from the user's description. "
        "4) Selected reference characteristics. "
        "5) Conservative musical defaults. "
        "Return each resolved constraint as a plain line: 'field: value (rationale)'. "
        "Do not use markdown formatting (no bullets, no bold/asterisks, no headers) "
        "and do not add any prose before or after the list of lines."
    )

    def build_prompt(self) -> str:
        brief = self.state.music_brief
        parts = []

        # Add explicit fields from the brief
        if brief.macro_genre:
            parts.append(f"Genre: {brief.macro_genre}")
        if brief.subgenres:
            parts.append(f"Subgenres: {', '.join(brief.subgenres)}")
        if brief.explicit_exclusions:
            parts.append(f"Exclusions: {', '.join(brief.explicit_exclusions)}")
        if brief.vocal_gender != "unspecified":
            parts.append(f"Vocal gender: {brief.vocal_gender}")
        if brief.tempo != "unspecified":
            parts.append(f"Tempo: {brief.tempo}")

        # Add section tags
        if self.state.explicit_sections:
            parts.append(f"Sections requested: {', '.join(self.state.explicit_sections)}")

        # Build prompt
        prompt = self.SYSTEM_PROMPT + "\n\n"
        prompt += "Music Brief fields:\n"
        for p in parts:
            prompt += f"- {p}\n"

        if not parts:
            prompt += "- No explicit fields detected\n"

        prompt += (
            "\nResolve conflicts and apply precedence rules. "
            "Return each constraint as 'Field: value (rationale)'."
        )
        return self._sanitize(prompt)

    def parse_response(self, response: str) -> CaptionState:
        constraints = {}
        for line in response.splitlines():
            match = re.match(r"(\w[\w\s]*)\s*[:=]\s*(.+?)(?:\s*\(.*?\))?\s*$", line)
            if match:
                field = match.group(1).strip().lower()
                value = match.group(2).strip()
                constraints[field] = value

        self.state.resolved_constraints = constraints
        return self.state

    async def run(self) -> CaptionState:
        """Execute the constraint resolution LLM call."""
        prompt = self.build_prompt()
        response = self._call_llm(prompt)
        return self.parse_response(response)
