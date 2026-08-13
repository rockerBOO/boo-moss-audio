"""Stage 3-4 — RouterAgent: Route to one or two families."""

from __future__ import annotations

import re

from music_caption.agents.base import BaseAgent
from ..models import CaptionState


class RouterAgent(BaseAgent):
    """Routes the Music Brief to one primary and optionally one secondary family."""

    SYSTEM_PROMPT = (
        "You are a genre router for MiniMax Music 3. "
        "Given a Music Brief, route it to one of these families: {families}. "
        "Rules:\n"
        "- Choose the primary family from the user's main genre, not from generic mood adjectives.\n"
        "- Add one secondary family only for explicit fusion or contrasting palette.\n"
        "- Read no more than two family indexes.\n"
        "- Treat ballad, emotional, epic, modern, dark, cinematic as modifiers unless stronger evidence exists.\n"
        "Return 'Primary: <family>' and optionally 'Secondary: <family>'."
    )

    def build_prompt(self) -> str:
        from ..router import GenreRouter

        router = GenreRouter()
        all_families = router.get_all_families()
        brief = self.state.music_brief

        # Build context about the brief
        parts = []
        if brief.macro_genre:
            parts.append(f"Macro genre: {brief.macro_genre}")
        if brief.subgenres:
            parts.append(f"Subgenres: {', '.join(brief.subgenres)}")
        if brief.cultural_style:
            parts.append(f"Cultural style: {brief.cultural_style}")
        if brief.groove:
            parts.append(f"Groove: {brief.groove}")
        if brief.core_instruments:
            parts.append(f"Instruments: {', '.join(brief.core_instruments)}")

        prompt = self.SYSTEM_PROMPT.format(families=", ".join(all_families))
        prompt += "\n\nMusic Brief:\n" + "\n".join(f"- {p}" for p in parts)
        return self._sanitize(prompt)

    def parse_response(self, response: str) -> CaptionState:
        primary = ""
        secondary = None

        # Extract primary family
        primary_match = re.search(r"[Pp]rimary\s*[:=]\s*(.+)", response)
        if primary_match:
            primary = primary_match.group(1).strip().lower()

        # Extract secondary family
        secondary_match = re.search(r"[Ss]econdary\s*[:=]\s*(.+)", response)
        if secondary_match:
            sec_val = secondary_match.group(1).strip().lower()
            if sec_val and sec_val not in ("none", "null", ""):
                secondary = sec_val

        self.state.primary_family = primary
        self.state.secondary_family = secondary
        return self.state

    async def run(self) -> CaptionState:
        """Execute the routing LLM call."""
        prompt = self.build_prompt()
        response = self._call_llm(prompt)
        return self.parse_response(response)
