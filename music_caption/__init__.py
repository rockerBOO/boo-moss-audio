"""BooMusicCaptionRewriter — orchestrator for the 8-stage agent pipeline."""

from __future__ import annotations

import asyncio
import re

from music_caption.models import CaptionState
from music_caption.agents.brief_agent import BriefAgent
from music_caption.agents.constraints_agent import ConstraintsAgent
from music_caption.agents.router_agent import RouterAgent
from music_caption.agents.selection_agent import SelectionAgent
from music_caption.agents.template_reader import TemplateReader
from music_caption.agents.timeline_agent import TimelineAgent
from music_caption.agents.renderer_agent import RendererAgent


def _extract_section_tags(lyrics: str) -> list[str]:
    """Extract bracketed section tags from lyrics text."""
    if not lyrics:
        return []
    return [tag.strip("[] ") for tag in re.findall(r"\[(.+?)\]", lyrics)]


class CaptionRewriter:
    """Orchestrates the 8-stage caption rewriter pipeline."""

    def __init__(self, llm_model: dict) -> None:
        self.llm_model = llm_model
        self.state = CaptionState()

    async def _run_stage(
        self,
        agent_cls,
        style_keywords: str,
        lyrics: str,
        stage_name: str,
    ) -> CaptionState:
        """Run a single agent stage with one retry on failure."""
        for attempt in range(2):
            try:
                state = await agent_cls(
                    self.llm_model, self.state,
                    style_keywords=style_keywords,
                    lyrics=lyrics,
                ).run()
                return state
            except Exception as exc:
                if attempt == 0:
                    continue
                self.state.stage_warnings.append(
                    f"{stage_name} failed after 2 attempts: {exc}"
                )
        return self.state

    async def rewrite(self, style_keywords: str, lyrics: str) -> str:
        """Run all 8 stages in sequence and return the final caption."""
        self.state.explicit_sections = _extract_section_tags(lyrics)

        self.state = await self._run_stage(
            BriefAgent, style_keywords, lyrics, "BriefAgent"
        )
        self.state = await self._run_stage(
            ConstraintsAgent, style_keywords, lyrics, "ConstraintsAgent"
        )
        self.state = await self._run_stage(
            RouterAgent, style_keywords, lyrics, "RouterAgent"
        )
        self.state = await self._run_stage(
            SelectionAgent, style_keywords, lyrics, "SelectionAgent"
        )

        try:
            self.state = await TemplateReader(
                self.llm_model, self.state,
                style_keywords=style_keywords,
                lyrics=lyrics,
            ).run()
        except Exception as exc:
            self.state.stage_warnings.append(f"TemplateReader failed: {exc}")

        self.state = await self._run_stage(
            TimelineAgent, style_keywords, lyrics, "TimelineAgent"
        )
        self.state = await self._run_stage(
            RendererAgent, style_keywords, lyrics, "RendererAgent"
        )

        caption = self.state.final_caption
        if self.state.stage_warnings:
            warnings_block = "\n".join(
                f"[Warning: {w}]" for w in self.state.stage_warnings
            )
            caption = f"{caption}\n\n{warnings_block}" if caption else warnings_block
        return caption
