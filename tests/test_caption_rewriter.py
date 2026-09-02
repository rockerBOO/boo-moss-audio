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

import pytest

# Use the same import path as the agents to ensure consistent module objects
from music_caption.models import (
    CaptionState,
    Confidence,
    AgentRole,
    IndexCard,
    SelectedReference,
)
from music_caption.router import GenreRouter
from music_caption.template_store import TemplateStore
from music_caption.agents.base import BaseAgent
from music_caption.agents.brief_agent import BriefAgent
from music_caption.agents.constraints_agent import ConstraintsAgent
from music_caption.agents.router_agent import RouterAgent
from music_caption.agents.selection_agent import SelectionAgent
from music_caption.agents.template_reader import TemplateReader
from music_caption.agents.timeline_agent import TimelineAgent
from music_caption.agents.renderer_agent import RendererAgent
from music_caption import CaptionRewriter, _extract_section_tags

# Root of the music_caption package for file I/O tests
PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "music_caption"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(responses: list[str]) -> dict:
    """Create a mock LLM dict that returns predetermined responses sequentially."""
    it = iter(responses)
    return {"generate": lambda *a, **k: next(it)}


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

    def test_get_cards_for_family_returns_real_cards(self) -> None:
        router = GenreRouter()
        cards = router.get_cards_for_family("east-asian-modern")
        assert len(cards) > 0
        assert all(card.template_path for card in cards)

    def test_family_map_has_no_markdown_artifacts(self) -> None:
        router = GenreRouter()
        router._load_all()
        assert router._family_map, "family map should not be empty"
        for route, index_name in router._family_map.items():
            assert "`" not in route
            assert "[" not in index_name and "(" not in index_name
            assert index_name in router._indexes

    def test_aliases_have_no_header_row(self) -> None:
        router = GenreRouter()
        router._load_all()
        assert "User wording" not in router._aliases

    def test_normalize_alias_known_term(self) -> None:
        router = GenreRouter()
        assert router.normalize_alias("华语流行、国语流行") == "Mandopop / C-pop"


class TestTemplateStore:
    def test_get_template_existing(self) -> None:
        store = TemplateStore()
        # Use a known template file from the bundled templates
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "templates"))
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
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "templates"))
        if len(sample_files) < 2:
            pytest.skip("Not enough templates for multi-file test")
        result = store.get_templates([sample_files[0], sample_files[1]])
        assert len(result) >= 2

    def test_clear_cache(self) -> None:
        store = TemplateStore()
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "templates"))
        if not sample_files:
            pytest.skip("No templates found")
        store.get_template(sample_files[0])
        assert len(store._cache) > 0
        store.clear_cache()
        assert len(store._cache) == 0

    def test_get_template_via_real_router_card(self) -> None:
        router = GenreRouter()
        cards = router.get_cards_for_family("east-asian-modern")
        assert cards, "expected at least one real card to test against"
        store = TemplateStore()
        content = store.get_template(cards[0].template_path)
        assert content is not None
        assert len(content) > 0


# ---------------------------------------------------------------------------
# BaseAgent._call_llm tests
# ---------------------------------------------------------------------------

class _EchoAgent(BaseAgent):
    """Minimal concrete BaseAgent for exercising _call_llm directly."""

    def build_prompt(self) -> str:
        return ""

    def parse_response(self, response: str) -> CaptionState:
        return self.state

    async def run(self) -> CaptionState:
        return self.state


class _FakeClip:
    """Duck-types ComfyUI's comfy.sd.CLIP: tokenize()/generate()/decode().

    Real CLIPLoader-wrapped models (e.g. Gemma, per comfy_extras/nodes_textgen.py)
    expose exactly this interface; BaseAgent._call_llm must detect and use it
    instead of treating llm_model as a dict.
    """

    def __init__(self, generated_text: str) -> None:
        self.generated_text = generated_text
        self.tokenize_calls: list[str] = []
        self.tokenize_kwargs: list[dict] = []
        self.generate_calls: list[dict] = []

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append(text)
        self.tokenize_kwargs.append(kwargs)
        return {"tokens": text}

    def generate(self, tokens, **kwargs):
        self.generate_calls.append(kwargs)
        return ["fake-token-ids"]

    def decode(self, token_ids, **kwargs):
        assert token_ids == ["fake-token-ids"]
        return self.generated_text


class TestCallLLM:
    def test_call_llm_uses_clip_like_object_when_present(self, caplog) -> None:
        state = _make_state()
        clip = _FakeClip("Macro genre: cinematic pop")
        agent = _EchoAgent(clip, state)

        with caplog.at_level("INFO"):
            result = agent._call_llm("some prompt")

        assert result == "Macro genre: cinematic pop"
        assert clip.tokenize_calls == ["some prompt"]
        assert len(clip.generate_calls) == 1
        # CLIP.generate defaults seed=None, which crashes
        # torch.Generator.manual_seed() whenever do_sample=True -- an
        # explicit int seed must always be passed.
        assert isinstance(clip.generate_calls[0]["seed"], int)
        # skip_template=False must be passed explicitly: Gemma's tokenizer
        # (comfy/text_encoders/lt.py's Gemma3_Tokenizer) defaults
        # skip_template=True, which drops chat-turn wrapping entirely.
        assert clip.tokenize_kwargs[0]["skip_template"] is False
        # Debugging: each agent's raw LLM response should be logged.
        assert "Macro genre: cinematic pop" in caplog.text
        assert "_EchoAgent" in caplog.text

    def test_call_llm_falls_back_to_prompt_when_llm_model_is_none(self, caplog) -> None:
        state = _make_state()
        agent = _EchoAgent(None, state)

        with caplog.at_level("INFO"):
            result = agent._call_llm("some prompt")

        assert result == "some prompt"
        # The no-LLM fallback isn't a real generation -- nothing to log.
        assert caplog.text == ""

    def test_call_llm_still_supports_dict_generate_callable(self) -> None:
        state = _make_state()
        mock_llm = _mock_llm(["dict-based response"])
        agent = _EchoAgent(mock_llm, state)

        assert agent._call_llm("some prompt") == "dict-based response"


class TestBriefAgent:
    def test_build_prompt_with_keywords(self) -> None:
        state = _make_state()
        agent = BriefAgent({}, state, style_keywords="deep house, emotional, arpeggios")
        prompt = agent.build_prompt()
        assert "deep house" in prompt

    def test_build_prompt_forces_exact_field_names_and_confidence_format(self) -> None:
        """Regression: a real Gemma run once used its own field labels
        ("Vocal Gender & Timbre") and bare-word confidence scores instead of
        the exact snake_case FIELDS names + parenthesized enum that
        parse_response requires, silently discarding every field. The prompt
        must spell out both the exact field names and the exact format.
        """
        state = _make_state()
        agent = BriefAgent({}, state, style_keywords="deep house")
        prompt = agent.build_prompt()

        for field in BriefAgent.FIELDS:
            assert field in prompt
        assert "EXPLICIT, TAGGED, INFERRED, UNSPECIFIED" in prompt
        assert "do not use markdown formatting" in prompt.lower()
        # SYSTEM_PROMPT previously wasn't referenced by build_prompt() at all
        # (a dead constant) -- confirm it's actually sent to the model.
        assert BriefAgent.SYSTEM_PROMPT in prompt

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

    def test_build_prompt_forbids_markdown_formatting(self) -> None:
        """Regression: a real Gemma run once wrapped its response in markdown
        bullets ("* **Genre: unspecified...**"), which parse_response's
        line-anchored regex (requires a line to start with \\w) can never
        match -- resolved_constraints came back empty despite real content.
        The prompt must explicitly forbid markdown so lines stay parseable.
        """
        state = _make_state()
        agent = ConstraintsAgent({}, state)
        prompt = agent.build_prompt()

        assert "do not use markdown formatting" in prompt.lower()
        assert "no bullets" in prompt.lower()
        # SYSTEM_PROMPT previously wasn't referenced by build_prompt() at all
        # (a dead constant) -- confirm it's actually sent to the model.
        assert ConstraintsAgent.SYSTEM_PROMPT in prompt

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
        state.primary_family = "pop"
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

    def test_parse_response_selects_matching_card_not_last(self) -> None:
        state = _make_state()
        state.primary_family = "pop"
        card_a = IndexCard(card_id="a", style="pop-a", template_path="pop_a.txt")
        card_b = IndexCard(card_id="b", style="pop-b", template_path="pop_b.txt")
        agent = SelectionAgent({}, state)

        original_get = GenreRouter.get_cards_for_family
        GenreRouter.get_cards_for_family = (
            lambda self, family: [card_a, card_b] if "pop" in family.lower() else []
        )

        try:
            response = "Role: Foundation\nTemplate: pop_a.txt\nRationale: best match\n"
            result = agent.parse_response(response)
            assert len(result.selected_references) == 1
            assert result.selected_references[0].card_id == "a"
        finally:
            GenreRouter.get_cards_for_family = original_get

    def test_parse_response_no_match_skips_reference(self) -> None:
        state = _make_state()
        state.primary_family = "pop"
        card_a = IndexCard(card_id="a", style="pop-a", template_path="pop_a.txt")
        agent = SelectionAgent({}, state)

        original_get = GenreRouter.get_cards_for_family
        GenreRouter.get_cards_for_family = (
            lambda self, family: [card_a] if "pop" in family.lower() else []
        )

        try:
            response = "Role: Foundation\nTemplate: nonexistent.txt\nRationale: no real match\n"
            result = agent.parse_response(response)
            assert result.selected_references == []
        finally:
            GenreRouter.get_cards_for_family = original_get


class TestTemplateReader:
    async def test_run_with_valid_templates(self) -> None:
        state = _make_state()
        sample_files = os.listdir(os.path.join(str(PACKAGE_ROOT), "templates"))
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
            "### Global Metadata\n"
            "A deep house track built on a hypnotic four-on-the-floor pulse at approximately 120 BPM "
            "with layered analog synthesizer pads, warm sub bass, and syncopated hi-hats creating a "
            "mesmerizing late-night club atmosphere that demands extended listening. The production "
            "foundation uses heavily filtered loops and subtle sidechain pumping, maintaining a minor-key "
            "harmonic bed throughout that stays understated and repetitive by deliberate design. Reverb-soaked "
            "percussion elements and a shuffled swing groove keep the energy rolling forward without ever "
            "feeling rushed or predictable, relying on small variations and textural shifts to carry the "
            "emotional arc through long DJ sets. The sonic palette favors organic warmth from aged equipment "
            "and vintage emulation, avoiding overly digital or synthetic tones that would break the hypnotic "
            "spell. Low-end emphasis on the sub bass creates a physical, tactile sensation that connects "
            "dancers to the groove's pulsing heartbeat, while filtered hi-hat layers add movement without "
            "harshness.\n\n"
            "### Vocal Details\n"
            "A solo female vocalist with a warm, slightly dusky timbre delivers soft, breathy phrases that "
            "sit deliberately low in the mix, functioning more as atmospheric texture than as lead melody. "
            "Her delivery maintains a relaxed, almost conversational quality throughout, avoiding belted "
            "power vocal techniques in favor of intimate proximity to the microphone. Light vocal chops and "
            "carefully placed stutter edits thread between the main synth hook, creating rhythmic punctuation "
            "without disrupting the groove. Occasional layered harmonies, voiced closely together for warmth, "
            "and airy ad-libs drift in during chorus sections, adding perceived depth without overpowering "
            "the carefully balanced arrangement. Reverb and delay on vocal tracks enhance the atmospheric "
            "quality rather than creating obvious spatial effects.\n\n"
            "### Arrangement\n"
            "Verse section opens with a stripped-back intro built from a lone filtered analog pad and a soft, "
            "cushioned kick pulse, gradually introducing filtered hi-hats and a rolling, slightly reedy sub "
            "bassline that locks into the groove. The first verse layers in deep resonant bass and soft "
            "crystalline pads, establishing the core groove's DNA while keeping the overall arrangement sparse "
            "and hypnotic, allowing individual elements to breathe. Pre-chorus builds tension through rising "
            "synth arpeggios and a subtle riser sweep. The chorus section brings the full arrangement online: "
            "driving groove, layered arpeggios with slight detuning for thickness, and the vocal hook locking "
            "together into a dense, danceable peak. A brief breakdown strips most elements away, leaving only "
            "bass and pads. Final chorus returns with added percussion layers and enhanced vocal stacks, "
            "closing the track on a sustained, heavily filtered outro that fades rather than stops.\n",
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

        mock_llm = {"generate": mock_generate}

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
            # Stage 8: Renderer — long enough (200-600 words) to pass validation on the
            # first attempt, so the pipeline calls the LLM exactly once per stage.
            "### Global Metadata\n"
            "A deep house track built around a driving four-on-the-floor pulse near 122 BPM, "
            "using analog synthesizer pads, warm sub bass, and syncopated hi-hats to create a "
            "hypnotic, late-night club atmosphere. The production favors filtered loops, subtle "
            "sidechain pumping, and a minor-key harmonic bed that stays understated and repetitive "
            "by design, letting small variations carry the emotional arc rather than dramatic key "
            "changes. Reverb-soaked percussion and a shuffled groove keep the energy rolling "
            "forward without ever feeling rushed.\n\n"
            "### Vocal Details\n"
            "A solo female vocalist delivers soft, breathy phrases that sit low in the mix, "
            "functioning more as texture than lead melody. Her delivery is relaxed and "
            "conversational, with light vocal chops and stutter edits threaded between the main "
            "synth hook. Occasional layered harmonies and airy ad-libs drift in during the chorus "
            "sections, adding warmth without ever overpowering the groove-driven arrangement.\n\n"
            "### Arrangement\n"
            "The track opens with a stripped-back intro built from a lone analog pad and a soft "
            "kick pulse, gradually introducing filtered hi-hats and a rolling sub bassline. The "
            "first verse layers in deep bass and soft pads, establishing the core groove while "
            "keeping the arrangement sparse and hypnotic. The pre-chorus adds rising synth "
            "arpeggios and a subtle riser to build anticipation. The chorus brings the full "
            "arrangement online: driving groove, layered arpeggios, and the vocal hook locking "
            "together into a dense, danceable peak. A brief breakdown strips elements away before "
            "the final chorus returns with added percussion layers, closing the track on a "
            "sustained, filtered outro.\n",
        ]

        def gen(*args, **kwargs):
            call_depth[0] += 1
            return responses[min(call_depth[0] - 1, len(responses) - 1)]

        mock_llm = {"generate": gen}

        rewriter = CaptionRewriter(mock_llm)
        result = asyncio.run(rewriter.rewrite("deep house, emotional, arpeggios", "[Verse] hello [Chorus] world"))

        assert "### Global Metadata" in result
        assert "### Vocal Details" in result
        assert "### Arrangement" in result
        assert call_depth[0] == 6  # Each stage called exactly once

    def test_empty_inputs(self) -> None:
        """Test that empty inputs produce a valid (minimal) caption."""
        mock_llm = {"generate": lambda *a, **k: ""}

        rewriter = CaptionRewriter(mock_llm)
        result = asyncio.run(rewriter.rewrite("", ""))
        assert isinstance(result, str)

    def test_section_tags_extracted(self) -> None:
        """Verify section tags are extracted and stored in state."""
        mock_llm = {"generate": lambda *a, **k: ""}

        rewriter = CaptionRewriter(mock_llm)
        asyncio.run(rewriter.rewrite("test", "[Verse] one [Chorus] two"))
        assert rewriter.state.explicit_sections == ["Verse", "Chorus"]


class TestStageWarnings:
    """Stage-failure warnings must survive to the final caption, not get overwritten."""

    async def test_run_stage_records_warning_after_two_failures(self) -> None:
        class AlwaysFailAgent(BaseAgent):
            def build_prompt(self) -> str:
                return ""

            def parse_response(self, response: str) -> CaptionState:
                return self.state

            async def run(self) -> CaptionState:
                raise RuntimeError("boom")

        rewriter = CaptionRewriter({})
        await rewriter._run_stage(AlwaysFailAgent, "kw", "", "FakeStage")
        assert rewriter.state.stage_warnings == ["FakeStage failed after 2 attempts: boom"]

    def test_warning_survives_when_later_stage_succeeds(self) -> None:
        """A BriefAgent failure must still show up in the final caption even though
        RendererAgent (the last stage) succeeds and writes its own final_caption."""
        call_count = [0]
        success_responses = [
            # ConstraintsAgent
            "genre: deep house (from user input)\n",
            # RouterAgent
            "Primary: deep-house\nSecondary: none\n",
            # SelectionAgent
            "Role: Foundation\nTemplate: deep-house_0001.txt\nRationale: best match\n",
            # TimelineAgent
            "[Verse] Deep bass and soft pads enter\n",
            # RendererAgent — long enough to pass validation on the first attempt
            "### Global Metadata\n" + ("Deep house track with analog synths. " * 20) + "\n\n"
            "### Vocal Details\n" + ("Soft layered female vocals. " * 20) + "\n\n"
            "### Arrangement\n" + ("Verse builds steadily into the chorus. " * 20) + "\n",
        ]

        def gen(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                # BriefAgent's two attempts both fail
                raise RuntimeError("boom")
            return success_responses[call_count[0] - 3]

        rewriter = CaptionRewriter({"generate": gen})
        result = asyncio.run(rewriter.rewrite("deep house", ""))

        assert "[Warning: BriefAgent failed after 2 attempts: boom]" in result
        assert "### Global Metadata" in result
