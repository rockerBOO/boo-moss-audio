"""Stage 6 — TemplateReader: Read selected templates from bundled files."""

from __future__ import annotations

from music_caption.agents.base import BaseAgent
from ..models import CaptionState


class TemplateReader(BaseAgent):
    """Reads each selected template file into the shared state."""

    def build_prompt(self) -> str:
        """TemplateReader has no LLM call; return a minimal prompt for consistency."""
        paths = [ref.template_path for ref in self.state.selected_references]
        if not paths:
            return "No references selected — nothing to read."
        return f"Read templates: {', '.join(paths)}"

    def parse_response(self, response: str) -> CaptionState:
        """Load each selected template into state.template_contents."""
        from ..template_store import TemplateStore

        store = TemplateStore()
        paths = [ref.template_path for ref in self.state.selected_references]
        loaded = store.get_templates(paths)

        # Merge into shared state
        self.state.template_contents.update(loaded)

        # Record warnings for missing templates
        for path in paths:
            if path not in loaded:
                # Try to fill style_description from the card's style field
                for ref in self.state.selected_references:
                    if ref.template_path == path and ref.card.style:
                        ref.style_description = ref.card.style
                        break

        return self.state

    async def run(self) -> CaptionState:
        """Execute file I/O to load selected templates."""
        if not self.state.selected_references:
            return self.state

        prompt = self.build_prompt()
        self.parse_response(prompt)
        return self.state
