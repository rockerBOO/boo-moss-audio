"""Stage 8 — RendererAgent: Render & validate final caption."""

from __future__ import annotations

import re

from music_caption.agents.base import BaseAgent
from ..models import CaptionState


class RendererAgent(BaseAgent):
    """Renders the final 3-heading caption with self-validation loop."""

    SYSTEM_PROMPT = (
        "You are a caption renderer for MiniMax Music 3. "
        "Produce a structured caption with exactly three headings in order: "
        "### Global Metadata, ### Vocal Details, ### Arrangement. "
        "Rules:\n"
        "- ~250-450 words total in English.\n"
        "- No quoted or paraphrased lyric content from the user's lyrics.\n"
        "- Do not fabricate exact BPM or key without evidence.\n"
        "- If the request is instrumental, state that explicitly; do not add vocals.\n"
        "- Each heading must contain specific musical descriptions, not generic labels.\n"
        "- Instruments should have coherent entrances, changes, and exits described.\n"
        "- The Arrangement section must follow a readable timeline with energy arc.\n"
        "- Keep it specific but concise — not an essay.\n"
        "Use all available context: Music Brief, resolved constraints, timeline plan, and selected references."
    )

    REQUIRED_HEADINGS = [
        "Global Metadata",
        "Vocal Details",
        "Arrangement",
    ]

    def build_prompt(self) -> str:
        brief = self.state.music_brief
        constraints = self.state.resolved_constraints
        timeline = self.state.timeline_plan
        refs = self.state.selected_references
        templates = self.state.template_contents
        sections = self.state.explicit_sections

        parts = []

        # Music Brief summary
        if brief.macro_genre:
            parts.append(f"Genre: {brief.macro_genre}")
        if brief.subgenres:
            parts.append(f"Subgenres: {', '.join(brief.subgenres)}")
        if brief.mood_arc:
            parts.append(f"Mood: {brief.mood_arc}")
        if brief.tempo != "unspecified":
            parts.append(f"Tempo: {brief.tempo}")
        if brief.vocal_presence != "unspecified":
            parts.append(f"Vocal presence: {brief.vocal_presence}")
        if brief.vocal_gender != "unspecified":
            parts.append(f"Vocal gender: {brief.vocal_gender}")
        if brief.core_instruments:
            parts.append(f"Instruments: {', '.join(brief.core_instruments)}")
        if brief.vocal_timbre:
            parts.append(f"Vocal timbre: {brief.vocal_timbre}")
        if brief.vocal_delivery:
            parts.append(f"Vocal delivery: {brief.vocal_delivery}")
        if brief.production_texture:
            parts.append(f"Production: {brief.production_texture}")
        if brief.groove:
            parts.append(f"Groove: {brief.groove}")
        if brief.meter != "unspecified":
            parts.append(f"Meter: {brief.meter}")
        if brief.cultural_style:
            parts.append(f"Cultural style: {brief.cultural_style}")
        if brief.spatial_character:
            parts.append(f"Spatial: {brief.spatial_character}")
        if brief.explicit_exclusions:
            parts.append(f"Exclusions: {', '.join(brief.explicit_exclusions)}")

        # Constraints
        if constraints:
            parts.append("\nConstraints:\n" + "\n".join(
                f"- {k}: {v}" for k, v in constraints.items()
            ))

        # Selected references summary
        if refs:
            parts.append("\nSelected references:\n")
            for ref in refs:
                desc = ref.style_description or ref.card.style or ref.card.compact_description
                parts.append(f"- [{ref.role.value}] {desc}")

        # Template contents preview
        if templates:
            parts.append("\nTemplate content:\n")
            for path, content in templates.items():
                preview = content[:300].replace("\n", " ")
                parts.append(f"- {path}: {preview}...")

        # Timeline plan
        if timeline:
            parts.append(f"\nTimeline plan:\n{timeline}")

        # Section tags
        if sections:
            parts.append(f"\nSections requested: {', '.join(sections)}")
        else:
            parts.append("\nNo explicit sections — use standard song structure.")

        prompt = self.SYSTEM_PROMPT + "\n\n" + "\n".join(parts)
        return self._sanitize(prompt)

    def _validate_caption(self, caption: str) -> list[str]:
        """Run self-validation checks on the caption. Returns list of failure reasons."""
        failures = []

        # Check all three required headings are present
        for heading in self.REQUIRED_HEADINGS:
            if f"### {heading}" not in caption:
                failures.append(f"Missing required heading: ### {heading}")

        # Check heading order
        heading_positions = []
        for heading in self.REQUIRED_HEADINGS:
            pos = caption.find(f"### {heading}")
            if pos >= 0:
                heading_positions.append((heading, pos))

        if len(heading_positions) == 3:
            if not all(
                heading_positions[i][1] < heading_positions[i + 1][1]
                for i in range(len(heading_positions) - 1)
            ):
                failures.append("Headings are not in the required order")

        # Check word count (250-450 words)
        word_count = len(caption.split())
        if word_count < 200 or word_count > 600:
            failures.append(f"Word count {word_count} is outside acceptable range (250-450)")

        # Check no quoted lyrics (simple heuristic: lines that look like lyric quotes)
        if self._lyrics:
            lyric_lines = [l.strip() for l in self._lyrics.split("\n") if l.strip()]
            for line in lyric_lines:
                if len(line) > 20 and f'"{line}"' in caption or f"'{line}'" in caption:
                    failures.append(f"Contains quoted lyric: {line[:40]}...")

        # Check instrumental constraint
        brief = self.state.music_brief
        if brief.vocal_presence == "instrumental":
            if re.search(r"\b(vocal|singing|vocals|voice)\b", caption, re.IGNORECASE):
                failures.append("Claims instrumental but contains vocal references")

        return failures

    def parse_response(self, response: str) -> CaptionState:
        """Store the final caption from the LLM."""
        self.state.final_caption = self._sanitize(response)
        return self.state

    async def run(self) -> CaptionState:
        """Execute rendering with self-validation and optional revision."""
        prompt = self.build_prompt()
        response = self._call_llm(prompt)

        # First pass: parse and validate
        self.parse_response(response)
        failures = self._validate_caption(self.state.final_caption)

        # If validation fails, try one revision
        if failures:
            revision_prompt = (
                f"Revise the caption to fix these issues: {'; '.join(failures)}\n\n"
                f"Current caption:\n{self.state.final_caption}\n\n"
                f"Use the same context but produce a corrected version."
            )
            revised = self._call_llm(revision_prompt)
            self.parse_response(revised)

        return self.state
