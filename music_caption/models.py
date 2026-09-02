"""Shared data classes for the caption rewriter agent pipeline."""

from __future__ import annotations

import enum


class Confidence(enum.Enum):
    EXPLICIT = "explicit"
    TAGGED = "tagged"
    INFERRED = "inferred"
    UNSPECIFIED = "unspecified"


class AgentRole(enum.Enum):
    FOUNDATION = "Foundation"
    MODIFIER = "Modifier"
    ARRANGEMENT = "Arrangement"


class MusicBrief:
    """Extracted musical intent from style keywords and optional lyrics."""

    def __init__(self) -> None:
        self.macro_genre: str = ""
        self.subgenres: list[str] = []
        self.cultural_style: str = ""
        self.mood_arc: str = ""
        self.tempo: str = "unspecified"
        self.meter: str = "unspecified"
        self.groove: str = ""
        self.vocal_presence: str = "unspecified"
        self.vocal_gender: str = "unspecified"
        self.vocal_register: str = ""
        self.vocal_timbre: str = ""
        self.vocal_delivery: str = ""
        self.core_instruments: list[str] = []
        self.production_texture: str = ""
        self.section_structure: list[str] = []
        self.spatial_character: str = ""
        self.explicit_exclusions: list[str] = []
        # Per-field confidence labels
        self.confidence: dict[str, Confidence] = {}

    def set_confidence(self, field: str, level: Confidence) -> None:
        self.confidence[field] = level

    def is_empty(self) -> bool:
        return (
            not self.macro_genre
            and not self.subgenres
            and not self.mood_arc
            and not self.core_instruments
        )


class IndexCard:
    """A compact style card parsed from a family index file."""

    def __init__(
        self,
        card_id: str,
        style: str,
        secondary_routes: list[str] | None = None,
        tempo_key: str = "",
        mood_arc: str = "",
        vocal_cue: str = "",
        core_palette: str = "",
        template_path: str = "",
    ) -> None:
        self.card_id = card_id
        self.style = style
        self.secondary_routes: list[str] = secondary_routes or []
        self.tempo_key = tempo_key
        self.mood_arc = mood_arc
        self.vocal_cue = vocal_cue
        self.core_palette = core_palette
        self.template_path = template_path

    @property
    def compact_description(self) -> str:
        """Return a single-line summary of this card for LLM context."""
        return (
            f"ID: {self.card_id} | Style: {self.style} | "
            f"Tempo/Key: {self.tempo_key} | Mood: {self.mood_arc} | "
            f"Vocal: {self.vocal_cue} | Core: {self.core_palette} | "
            f"Template: {self.template_path}"
        )


class SelectedReference:
    """A selected reference template with its role."""

    def __init__(
        self,
        role: AgentRole,
        card: IndexCard,
        style_description: str,
    ) -> None:
        self.role = role
        self.card = card
        self.style_description = style_description

    @property
    def template_path(self) -> str:
        return self.card.template_path

    @property
    def card_id(self) -> str:
        return self.card.card_id


class CaptionState:
    """Shared state that flows through all agent stages."""

    def __init__(self) -> None:
        self.music_brief = MusicBrief()
        self.resolved_constraints: dict = {}
        self.primary_family: str = ""
        self.secondary_family: str | None = None
        self.selected_references: list[SelectedReference] = []
        self.template_contents: dict[str, str] = {}
        self.timeline_plan: str = ""
        self.final_caption: str = ""
        # Track which sections were explicitly requested from lyrics
        self.explicit_sections: list[str] = []
        # Warnings recorded when a stage fails but the pipeline continues
        self.stage_warnings: list[str] = []
