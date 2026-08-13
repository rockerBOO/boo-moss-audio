"""Stage 7 — TimelineAgent: Plan section-by-section timeline."""

from __future__ import annotations

from music_caption.agents.base import BaseAgent
from ..models import CaptionState


class TimelineAgent(BaseAgent):
    """Plans a coherent section-by-section timeline with energy arc."""

    SYSTEM_PROMPT = (
        "You are a timeline planner for MiniMax Music 3. "
        "Given a Music Brief, resolved constraints, selected templates, and optional section tags, "
        "design a coherent section-by-section timeline. "
        "Each section must describe what enters, exits, changes, or intensifies. "
        "Include instrument lifecycle, groove development, transitions, embellishments, texture, and spatial effects. "
        "Create a readable energy arc across the song. "
        "Do not produce static equipment lists — each line must describe change or motion."
    )

    def build_prompt(self) -> str:
        brief = self.state.music_brief
        constraints = self.state.resolved_constraints
        templates = self.state.template_contents
        sections = self.state.explicit_sections

        parts = []

        # Music Brief summary
        if brief.macro_genre:
            parts.append(f"Genre: {brief.macro_genre}")
        if brief.subgenres:
            parts.append(f"Subgenres: {', '.join(brief.subgenres)}")
        if brief.mood_arc:
            parts.append(f"Mood arc: {brief.mood_arc}")
        if brief.tempo != "unspecified":
            parts.append(f"Tempo: {brief.tempo}")
        if brief.vocal_presence != "unspecified":
            parts.append(f"Vocal presence: {brief.vocal_presence}")
        if brief.core_instruments:
            parts.append(f"Core instruments: {', '.join(brief.core_instruments)}")

        # Constraints
        if constraints:
            parts.append("\nResolved constraints:\n" + "\n".join(
                f"- {k}: {v}" for k, v in constraints.items()
            ))

        # Template characteristics (summarize each selected reference)
        if self.state.selected_references:
            parts.append("\nSelected references:\n")
            for ref in self.state.selected_references:
                parts.append(f"- [{ref.role.value}] {ref.card.style} ({ref.template_path})")

        # Template contents preview (first 200 chars each)
        if templates:
            parts.append("\nTemplate excerpts:\n")
            for path, content in templates.items():
                preview = content[:200].replace("\n", " ")
                parts.append(f"- {path}: {preview}...")

        # Section tags from lyrics
        if sections:
            parts.append(f"\nSections requested: {', '.join(sections)}")
        else:
            parts.append("\nNo explicit sections — use a standard song structure.")

        prompt = self.SYSTEM_PROMPT + "\n\n" + "\n".join(parts)
        return self._sanitize(prompt)

    def parse_response(self, response: str) -> CaptionState:
        """Store the raw timeline plan from the LLM."""
        self.state.timeline_plan = self._sanitize(response)
        return self.state

    async def run(self) -> CaptionState:
        """Execute the timeline planning LLM call."""
        prompt = self.build_prompt()
        response = self._call_llm(prompt)
        self.parse_response(response)
        return self.state
