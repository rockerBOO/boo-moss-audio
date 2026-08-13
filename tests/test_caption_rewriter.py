"""Tests for the BooMusicCaptionRewriter node pipeline.

Covers each agent's parse_response / build_prompt logic, the
TemplateReader file I/O, and a full-pipeline integration test with
a mock LLM that returns predetermined responses at each stage.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from boo_moss_audio.music_caption.models import (
    CaptionState,
    Confidence,
    AgentRole,
    IndexCard,
    SelectedReference,
)
from boo_moss_audio.music_caption.router import GenreRouter
from boo_moss_audio.music_caption.template_store import TemplateStore
from boo_moss_audio.music_caption.agents.brief_agent import BriefAgent
from boo_moss_audio.music_caption.agents.constraints_agent import ConstraintsAgent
from boo_moss_audio.music_caption.agents.router_agent import RouterAgent
from boo_moss_audio.music_caption.agents.selection_agent import SelectionAgent
from boo_moss_audio.music_caption.agents.template_reader import TemplateReader
from boo_moss_audio.music_caption.agents.timeline_agent import TimelineAgent
from boo_moss_audio.music_caption.agents.renderer_agent import RendererAgent
from boo_moss_audio.music_caption import CaptionRewriter, _extract_section_tags

# Root of the music_caption package for file I/O tests
PACKAGE_ROOT = Path(__file__).resolve().parent / "music_caption"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(responses: list[str]) -> dict:
    """Create a mock LLM dict that returns predetermined responses sequentially."""
    return {"generate": lambda *a, **k: next(iter(responses))}


class MockLLM:
    """Return predetermined responses for each LLM call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, *args, **kwargs) -> str:
        return next(self.responses)


def _make_state() -> CaptionState:
    return CaptionState()


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestExtractSectionTags:
    """Test the _extract_section_tags helper."""

    def test_empty_lyrics(self) -> None:
        assert _extract_section_tags("") == []
        assert _extract_section_tags(None) == []

    def test_bracketed_tags(self) -> None:
        lyrics = "[Verse] Hello world [Chorus] Singing now"
        assert _extract_section_tags(lyrics) == ["Verse", "Chorus"]

    def test_multiple_same_tag(self) -> None:
        lyrics = "[Verse] one [Verse] two [Bridge] three"
        assert _extract_section_tags(lyrics) == ["Verse", "Verse", "Bridge"]


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_all_values_present(self) -> None:
        values = {e.value for e in Confidence}
        assert values == {"explicit", "tagged", "inferred", "unspecified"}


class TestMusicBrief:
    def test_default_values(self) -> None:
        from boo_moss_audio.music_caption.models import MusicBrief

        brief = MusicBrief()
        assert brief.macro_genre == ""
        assert brief.subgenres == []
        assert brief.tempo == "unspecified"
        assert brief.vocal_presence == "unspecified"
        assert brief.is_empty() is True

    def test_set_confidence(self) -> None:
        from boo_moss_audio.music_caption.models import MusicBrief

        brief = MusicBrief()
        brief.set_confidence("tempo", Confidence.EXPLICIT)
        assert brief.confidence["tempo"] == Confidence.EXPLICIT


class TestIndexCard:
    def test_compact_description(self) -> None:
        card = IndexCard(
            card_id="test-001",
            style="deep house",
            tempo_key="120 BPM / Am",
            mood_arc="driving, emotional",
            vocal_cue="",
            core_palette="analog synth, deep bass",
            template_path="deep-house_0001.txt",
        )
        desc = card.compact_description
        assert "test-001" in desc
        assert "deep house" in desc
        assert "deep-house_0001.txt" in desc


class TestSelectedReference:
    def test_properties(self) -> None:
        card = IndexCard(
            card_id="t1",
            style="pop",
            tempo_key="",
            mood_arc="",
            vocal_cue="",
            core_palette="",
            template_path="pop_0001.txt",
        )
        ref = SelectedReference(AgentRole.FOUNDATION, card, "pop description")
        assert ref.role == AgentRole.FOUNDATION
        assert ref.template_path == "pop_0001.txt"
        assert ref.card_id == "t1"


class TestCaptionState:
    def test_default_state(self) -> None:
        state = CaptionState()
        assert state.music_brief.is_empty() is True
        assert state.primary_family == ""
        assert state.secondary_family is None
        assert state.selected_references == []
        assert state.template_contents == {}
        assert state.final_caption == ""


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestGenreRouter:
    def test_get_all_families(self) -> None:
        router = GenreRouter()
        families = router.get_all_families()
        assert len(families) >= 10  # Should have multiple families

    def test_get_cards_for_family(self) -> None:
        router = GenreRouter()
        families = router.get_all_families()
        for family in families[:3]:
            cards = router.get_cards_for_family(family)
            assert isinstance(cards, list)

    def test_normalize_alias(self) -> None:
        router = GenreRouter()
        # Unknown term passes through unchanged
        assert router.normalize_alias("unknown_thing_xyz") == "unknown_thing_xyz"


class TestTemplateStore:
    def test_get_template_existing(self) -> None:
        store = TemplateStore()
        # Use a known template file from the bundled templates
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "music_caption", "templates"))
        if not sample_files:
            pytest.skip("No templates found")
        first_file = sample_files[0]
        content = store.get_template(first_file)
        assert content is not None
        assert len(content) > 0

    def test_get_template_missing(self) -> None:
        store = TemplateStore()
        result = store.get_template("nonexistent_file_12345.txt")
        assert result is None
        assert len(store.warnings) > 0

    def test_get_templates_multiple(self) -> None:
        store = TemplateStore()
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "music_caption", "templates"))
        if len(sample_files) < 2:
            pytest.skip("Not enough templates for multi-file test")
        result = store.get_templates([sample_files[0], sample_files[1]])
        assert len(result) >= 2

    def test_clear_cache(self) -> None:
        store = TemplateStore()
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "music_caption", "templates"))
        if not sample_files:
            pytest.skip("No templates found")
        store.get_template(sample_files[0])
        assert len(store._cache) > 0
        store.clear_cache()
        assert len(store._cache) == 0


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestBriefAgent:
    def test_build_prompt_with_keywords(self) -> None:
        state = _make_state()
        agent = BriefAgent({}, state, style_keywords="deep house, emotional, arpeggios")
        prompt = agent.build_prompt()
        assert "deep house" in prompt

    def test_parse_response(self) -> None:
        state = _make_state()
        agent = BriefAgent({}, state)
        response = (
            "macro_genre: deep house\n"
            "subgenres: deep house, electronic\n"
            "mood_arc: driving, emotional\n"
            "tempo: 120 BPM\n"
            "vocal_presence: background\n"
            "core_instruments: analog synth, deep bass\n"
        )
        state = agent.parse_response(response)
        assert state.music_brief.macro_genre == "deep house"
        assert "deep house" in state.music_brief.subgenres
        assert state.music_brief.tempo == "120 BPM"

    async def test_run_with_mock_llm(self) -> None:
        state = _make_state()
        mock_llm = _mock_llm([
            "macro_genre: jazz fusion\nsubgenres: jazz, fusion\n"
            "tempo: fast\nvocal_presence: instrumental\n"
            "core_instruments: saxophone, electric piano\n",
        ])
        agent = BriefAgent(mock_llm, state, style_keywords="jazz fusion")
        await agent.run()
        assert state.music_brief.macro_genre == "jazz fusion"


class TestConstraintsAgent:
    def test_build_prompt_with_constraints(self) -> None:
        state = _make_state()
        state.music_brief.macro_genre = "deep house"
        state.music_brief.explicit_exclusions = ["vocals", "piano"]
        state.explicit_sections = ["Verse", "Chorus"]
        agent = ConstraintsAgent({}, state, style_keywords="deep house")
        prompt = agent.build_prompt()
        assert "deep house" in prompt
        assert "Exclusions" in prompt

    def test_parse_response(self) -> None:
        state = _make_state()
        agent = ConstraintsAgent({}, state)
        response = (
            "genre: deep house (from user input)\n"
            "vocal gender: female (inferred from timbre)\n"
            "tempo: 120 BPM max (from constraints)\n"
        )
        result = agent.parse_response(response)
        assert result.resolved_constraints["genre"] == "deep house"
        assert result.resolved_constraints["vocal gender"] == "female"  # rationale stripped


class TestRouterAgent:
    def test_build_prompt(self) -> None:
        state = _make_state()
        state.music_brief.macro_genre = "cinematic"
        state.music_brief.subgenres = ["orchestral", "ambient"]
        agent = RouterAgent({}, state, style_keywords="cinematic orchestral")
        prompt = agent.build_prompt()
        assert "cinematic" in prompt.lower() or "orchestral" in prompt.lower()

    def test_parse_response_primary(self) -> None:
        state = _make_state()
        agent = RouterAgent({}, state)
        response = "Primary: east-asian-modern\nSecondary: none"
        result = agent.parse_response(response)
        assert result.primary_family == "east-asian-modern"
        assert result.secondary_family is None

    def test_parse_response_with_secondary(self) -> None:
        state = _make_state()
        agent = RouterAgent({}, state)
        response = "Primary: pop\nSecondary: electronic"
        result = agent.parse_response(response)
        assert result.primary_family == "pop"
        assert result.secondary_family == "electronic"


class TestSelectionAgent:
    def test_build_prompt_no_cards(self) -> None:
        state = _make_state()
        state.primary_family = "nonexistent-family-xyz"
        agent = SelectionAgent({}, state)
        prompt = agent.build_prompt()
        assert "No cards available" in prompt

    def test_parse_response(self) -> None:
        state = _make_state()
        card = IndexCard(
            card_id="t1",
            style="pop",
            tempo_key="",
            mood_arc="",
            vocal_cue="",
            core_palette="",
            template_path="pop_0001.txt",
        )
        state.selected_references = []
        agent = SelectionAgent({}, state)

        # Manually inject a card into the router's cache
        from boo_moss_audio.music_caption.router import GenreRouter
        original_get = GenreRouter.get_cards_for_family

        def mock_get(self, family):
            if "pop" in family.lower():
                return [card]
            return []

        GenreRouter.get_cards_for_family = mock_get

        try:
            response = (
                "Role: Foundation\nTemplate: pop_0001.txt\nRationale: best match\n"
            )
            result = agent.parse_response(response)
            assert len(result.selected_references) == 1
            assert result.selected_references[0].role == AgentRole.FOUNDATION
        finally:
            GenreRouter.get_cards_for_family = original_get


class TestTemplateReader:
    async def test_run_with_valid_templates(self) -> None:
        state = _make_state()
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "music_caption", "templates"))
        if not sample_files:
            pytest.skip("No templates found")

        card = IndexCard(
            card_id="t1",
            style="test",
            template_path=sample_files[0],
        )
        state.selected_references = [SelectedReference(AgentRole.FOUNDATION, card, "")]

        agent = TemplateReader({}, state)
        await agent.run()
        assert len(state.template_contents) >= 1

    async def test_run_no_references(self) -> None:
        state = _make_state()
        agent = TemplateReader({}, state)
        result = await agent.run()
        assert result.template_contents == {}


class TestTimelineAgent:
    def test_build_prompt(self) -> None:
        state = _make_state()
        state.music_brief.macro_genre = "deep house"
        state.explicit_sections = ["Verse", "Chorus", "Bridge"]
        agent = TimelineAgent({}, state, style_keywords="deep house")
        prompt = agent.build_prompt()
        assert "deep house" in prompt.lower()
        assert "Verse" in prompt

    async def test_run_with_mock_llm(self) -> None:
        state = _make_state()
        mock_llm = _mock_llm([
            "[Verse] Start with deep bass and soft pads\n"
            "[Chorus] Full arrangement kicks in\n"
            "[Bridge] Minimal texture returns\n",
        ])
        agent = TimelineAgent(mock_llm, state)
        await agent.run()
        assert state.timeline_plan.strip() != ""


class TestRendererAgent:
    def test_build_prompt(self) -> None:
        state = _make_state()
        state.music_brief.macro_genre = "pop"
        state.timeline_plan = "[Verse] ... [Chorus] ..."
        agent = RendererAgent({}, state, style_keywords="pop")
        prompt = agent.build_prompt()
        assert "pop" in prompt.lower()

    def test_validate_caption_all_headings_present(self) -> None:
        state = _make_state()
        agent = RendererAgent({}, state)
        caption = (
            "### Global Metadata\nSome metadata here.\n\n"
            "### Vocal Details\nVocal info here.\n\n"
            "### Arrangement\nArrangement details.\n"
        )
        failures = agent._validate_caption(caption)
        heading_failures = [f for f in failures if "heading" in f.lower() or "Heading" in f]
        assert len(heading_failures) == 0, f"Unexpected heading failures: {heading_failures}"

    def test_validate_caption_missing_heading(self) -> None:
        state = _make_state()
        agent = RendererAgent({}, state)
        caption = "### Global Metadata\nOnly one section.\n"
        failures = agent._validate_caption(caption)
        assert any("Vocal Details" in f for f in failures)

    def test_validate_instrumental_vocal_conflict(self) -> None:
        state = _make_state()
        state.music_brief.vocal_presence = "instrumental"
        agent = RendererAgent({}, state)
        caption = "### Global Metadata\nInstrumental track.\n\n" \
                  "### Vocal Details\nBright female vocals with reverb.\n\n" \
                  "### Arrangement\nFull arrangement.\n"
        failures = agent._validate_caption(caption)
        assert any("instrumental" in f.lower() for f in failures)

    async def test_run_with_mock_llm(self) -> None:
        state = _make_state()
        mock_llm = _mock_llm([
            "### Global Metadata\nDeep house genre, 120 BPM.\n\n"
            "### Vocal Details\nSoft female vocals, dreamy.\n\n"
            "### Arrangement\nVerse starts minimal, chorus builds full.\n",
        ])
        agent = RendererAgent(mock_llm, state)
        await agent.run()
        assert "### Global Metadata" in state.final_caption
        assert "### Vocal Details" in state.final_caption
        assert "### Arrangement" in state.final_caption

    async def test_run_revision_on_failure(self) -> None:
        """Test that renderer revises once when validation fails."""
        state = _make_state()
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: missing headings
                return "Only one section\n### Global Metadata\nText here.\n"
            # Second call (revision): all good
            return (
                "### Global Metadata\nMetadata text.\n\n"
                "### Vocal Details\nVocal info.\n\n"
                "### Arrangement\nArrangement details.\n"
            )

        mock_llm = MagicMock()
        mock_llm.generate = mock_generate

        agent = RendererAgent(mock_llm, state)
        await agent.run()
        assert call_count == 2  # First attempt + one revision
        assert "### Global Metadata" in state.final_caption
        assert "### Vocal Details" in state.final_caption
        assert "### Arrangement" in state.final_caption


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestCaptionRewriterIntegration:
    """End-to-end test of the full pipeline with a mock LLM."""

    def test_full_pipeline(self) -> None:
        """Run the full CaptionRewriter with a mock LLM and verify output structure."""
        call_depth = [0]
        # Responses for each stage in order:
        # Stage 1 (BriefAgent), Stage 2 (ConstraintsAgent), Stage 3-4 (RouterAgent),
        # Stage 5 (SelectionAgent), Stage 7 (TimelineAgent, TemplateReader has no LLM),
        # Stage 8 (RendererAgent)
        responses = [
            # Stage 1: Brief
            "macro_genre: deep house\nsubgenres: deep house, electronic\n"
            "mood_arc: driving\nvocal_presence: background\ncore_instruments: analog synth, deep bass\n",
            # Stage 2: Constraints
            "genre: deep house (from user input)\nvocal gender: female (inferred)\n",
            # Stage 3-4: Router
            "Primary: deep-house\nSecondary: none\n",
            # Stage 5: Selection
            "Role: Foundation\nTemplate: deep-house_0001.txt\nRationale: best match\n",
            # Stage 7: Timeline
            "[Verse] Deep bass and soft pads enter\n[Chorus] Full arrangement with synth arpeggios\n",
            # Stage 8: Renderer
            "### Global Metadata\nDeep house genre at moderate tempo with analog synths.\n\n"
            "### Vocal Details\nSoft female vocals layered behind the mix.\n\n"
            "### Arrangement\nVerse opens with deep bass and soft pads. Chorus brings full synth arpeggios and driving groove.\n",
        ]

        mock_llm = MagicMock()
        def gen(*args, **kwargs):
            call_depth[0] += 1
            return responses[min(call_depth[0] - 1, len(responses) - 1)]
        mock_llm.generate = gen

        rewriter = CaptionRewriter(mock_llm)
        result = asyncio.run(rewriter.rewrite("deep house, emotional, arpeggios", "[Verse] hello [Chorus] world"))

        assert "### Global Metadata" in result
        assert "### Vocal Details" in result
        assert "### Arrangement" in result
        assert call_depth[0] == 6  # Each stage called exactly once

    def test_empty_inputs(self) -> None:
        """Test that empty inputs produce a valid (minimal) caption."""
        mock_llm = MagicMock()
        mock_llm.generate = lambda *a, **k: ""

        rewriter = CaptionRewriter(mock_llm)
        result = asyncio.run(rewriter.rewrite("", ""))
        assert isinstance(result, str)

    def test_section_tags_extracted(self) -> None:
        """Verify section tags are extracted and stored in state."""
        mock_llm = MagicMock()
        mock_llm.generate = lambda *a, **k: ""

        rewriter = CaptionRewriter(mock_llm)
        asyncio.run(rewriter.rewrite("test", "[Verse] one [Chorus] two"))
        assert rewriter.state.explicit_sections == ["Verse", "Chorus"]
