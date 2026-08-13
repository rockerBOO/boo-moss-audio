"""Genre router — loads and parses all family index files."""

from __future__ import annotations

import os
import re
from pathlib import Path

from music_caption.models import IndexCard


class GenreRouter:
    """Loads genre-router.md and all index files, parses compact cards."""

    def __init__(self, base_dir: str | None = None) -> None:
        if base_dir is None:
            base_dir = os.path.dirname(__file__)
        self.base_dir = base_dir
        self._router_text: str | None = None
        self._family_map: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._indexes: dict[str, list[IndexCard]] = {}
        self._loaded = False

    def _load_all(self) -> None:
        """Load genre-router.md and all index-*.md files."""
        if self._loaded:
            return

        # Load genre router
        router_path = os.path.join(self.base_dir, "references", "genre-router.md")
        with open(router_path, "r", encoding="utf-8") as f:
            self._router_text = f.read()
        self._parse_router(self._router_text)

        # Load all index files
        refs_dir = os.path.join(self.base_dir, "references")
        for filename in sorted(os.listdir(refs_dir)):
            if filename.startswith("index-") and filename.endswith(".md"):
                filepath = os.path.join(refs_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self._indexes[filename] = self._parse_index(content, filename)

        self._loaded = True

    def _parse_router(self, text: str) -> None:
        """Parse the family map and alias tables from genre-router.md."""
        self._family_map = self._parse_table(text, "## Family map", key_col=0, value_col=3)
        self._aliases = self._parse_table(text, "## Common aliases", key_col=0, value_col=1)

    def _parse_table(
        self, text: str, section_header: str, key_col: int, value_col: int
    ) -> dict[str, str]:
        """Parse a markdown table under `section_header` into a key/value dict.

        Skips the header row and the `|---` separator row — only rows after
        the separator are treated as data.
        """
        result: dict[str, str] = {}
        in_section = False
        table_started = False
        for line in text.splitlines():
            if section_header in line:
                in_section = True
                table_started = False
                continue
            if in_section and line.startswith("## "):
                break
            if not in_section:
                continue
            if line.startswith("|---") or line.startswith("| ---"):
                table_started = True
                continue
            if not table_started or not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) <= max(key_col, value_col):
                continue
            key = parts[key_col].strip("`")
            value = self._extract_link_target(parts[value_col])
            result[key] = value
        return result

    @staticmethod
    def _extract_link_target(text: str) -> str:
        """Extract the target from a markdown link `[text](target)`, else return the text as-is."""
        match = re.search(r"\(([^)]+)\)", text)
        if match:
            return match.group(1)
        return text.strip("`")

    def _parse_index(self, content: str, filename: str) -> list[IndexCard]:
        """Parse compact cards from an index file.

        Each row in the markdown table is a card. Format:
        | ID | Style | Secondary routes | Tempo / key | Mood arc | Vocal cue | Core palette | Template |
        """
        cards = []
        in_table = False
        header_parsed = False

        for line in content.splitlines():
            if not line.startswith("|"):
                continue

            # Detect table header
            if "|---" in line or "| ---" in line:
                if not header_parsed:
                    header_parsed = True
                in_table = True
                continue

            if not in_table:
                continue

            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 8:
                continue

            card_id = parts[0].strip("`")
            style = parts[1]
            secondary_routes_str = parts[2]
            tempo_key = parts[3]
            mood_arc = parts[4]
            vocal_cue = parts[5]
            core_palette = parts[6]
            template_path = parts[7].strip("`")

            # Parse secondary routes
            if secondary_routes_str == "—":
                secondary_routes: list[str] = []
            else:
                secondary_routes = [r.strip() for r in secondary_routes_str.split(",")]

            cards.append(IndexCard(
                card_id=card_id,
                style=style,
                secondary_routes=secondary_routes,
                tempo_key=tempo_key,
                mood_arc=mood_arc,
                vocal_cue=vocal_cue,
                core_palette=core_palette,
                template_path=template_path,
            ))

        return cards

    @property
    def all_cards(self) -> list[IndexCard]:
        """Return all cards from all families."""
        self._load_all()
        all_cards = []
        for cards in self._indexes.values():
            all_cards.extend(cards)
        return all_cards

    def get_cards_for_family(self, family: str) -> list[IndexCard]:
        """Return all cards for a given family name.

        The family name should match the route key (e.g. 'east-asian-modern').
        Maps to index-east-asian-modern.md etc.
        """
        self._load_all()
        index_name = self._family_map.get(family)
        if not index_name:
            return []
        return self._indexes.get(index_name, [])

    def get_family_index(self, family: str) -> str | None:
        """Get the index filename for a given family."""
        self._load_all()
        return self._family_map.get(family)

    def normalize_alias(self, term: str) -> str:
        """Normalize a term using known aliases. Returns the normalized form."""
        self._load_all()
        # Direct match
        if term in self._aliases:
            return self._aliases[term]
        # Case-insensitive match
        for alias, normalized in self._aliases.items():
            if alias.lower() == term.lower():
                return normalized
        return term

    def get_all_families(self) -> list[str]:
        """Return all known family route names."""
        self._load_all()
        return list(self._family_map.keys())
