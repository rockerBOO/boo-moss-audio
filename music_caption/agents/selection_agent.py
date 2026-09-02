"""Stage 5 — SelectionAgent: Select up to 3 references with distinct roles."""

from __future__ import annotations

import re

from .base import BaseAgent
from ..models import AgentRole, CaptionState, SelectedReference


class SelectionAgent(BaseAgent):
    """Selects up to 3 references with distinct roles."""

    _ROLES = [
        ("Foundation", "closest overall identity, groove, and songwriting language"),
        ("Modifier", "best source for a requested secondary genre, vocal character, cultural color, or production texture"),
        ("Arrangement", "best source for section development, energy contour, transitions, and instrument lifecycle"),
    ]

    def build_prompt(self) -> str:
        from ..router import GenreRouter

        router = GenreRouter()
        state = self.state

        # Get cards from selected families
        all_cards = []
        family = state.primary_family
        if family:
            cards = router.get_cards_for_family(family)
            all_cards.extend(cards)

        secondary = state.secondary_family
        if secondary:
            cards = router.get_cards_for_family(secondary)
            all_cards.extend(cards)

        # Build prompt with all cards
        prompt = (
            "You are a reference selector for MiniMax Music 3.\n"
            "Select up to 3 references from the following cards with distinct roles:\n"
            "- Foundation: closest overall identity, groove, and songwriting language\n"
            "- Modifier: best source for secondary genre, vocal character, cultural color, or production texture\n"
            "- Arrangement: best source for section development, energy contour, transitions\n\n"
            "Priority order: genre compatibility > explicit constraints > groove/tempo > vocal config > instrumentation > mood > production.\n\n"
            "Music Brief:\n"
        )

        brief = state.music_brief
        if brief.macro_genre:
            prompt += f"- Genre: {brief.macro_genre}\n"
        if brief.subgenres:
            prompt += f"- Subgenres: {', '.join(brief.subgenres)}\n"
        if brief.mood_arc:
            prompt += f"- Mood: {brief.mood_arc}\n"
        if brief.vocal_gender != "unspecified":
            prompt += f"- Vocal gender: {brief.vocal_gender}\n"
        if brief.core_instruments:
            prompt += f"- Instruments: {', '.join(brief.core_instruments)}\n"

        prompt += "\n\nAvailable cards:\n"
        for card in all_cards:
            prompt += f"{card.compact_description}\n"

        if not all_cards:
            prompt += "(No cards available for selected families)\n"

        prompt += (
            "\nReturn each selection as 'Role: <role>\nTemplate: <template_path>\nRationale: <reason>'\n"
            "Use one or two references when the request is simple."
        )
        return self._sanitize(prompt)

    @staticmethod
    def _find_card(cards, template_path: str):
        for card in cards:
            if card.template_path == template_path:
                return card
        return None

    def parse_response(self, response: str) -> CaptionState:
        state = self.state
        state.selected_references = []

        # Parse each selection block
        blocks = re.split(r"(?=Role:\s*[-=])", response, flags=re.IGNORECASE)
        role_map = {
            "foundation": AgentRole.FOUNDATION,
            "modifier": AgentRole.MODIFIER,
            "arrangement": AgentRole.ARRANGEMENT,
        }

        from ..router import GenreRouter
        router = GenreRouter()

        for block in blocks:
            if not re.search(r"[Rr]ole\s*[:=]", block):
                continue

            role_match = re.search(r"[Rr]ole\s*[:=]\s*(\w+)", block)
            template_match = re.search(r"[Tt]emplate\s*[:=]\s*(.+)", block)

            if not role_match or not template_match:
                continue

            role_str = role_match.group(1).lower()
            if role_str not in role_map:
                continue

            role = role_map[role_str]
            template_path = template_match.group(1).strip()

            # Find the card for this template
            card = self._find_card(
                router.get_cards_for_family(state.primary_family), template_path
            )
            if card is None and state.secondary_family:
                card = self._find_card(
                    router.get_cards_for_family(state.secondary_family), template_path
                )

            if card is None:
                continue

            state.selected_references.append(SelectedReference(
                role=role,
                card=card,
                style_description="",  # will be filled from template
            ))

        return self.state

    async def run(self) -> CaptionState:
        """Execute the reference selection LLM call."""
        prompt = self.build_prompt()
        response = self._call_llm(prompt)
        return self.parse_response(response)
