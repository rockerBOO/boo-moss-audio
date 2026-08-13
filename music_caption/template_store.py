"""Template store — lazy-loads caption templates from bundled files."""

from __future__ import annotations

import os
from pathlib import Path


class TemplateStore:
    """Loads and caches template .txt files from the bundled templates/ directory."""

    def __init__(self, base_dir: str | None = None) -> None:
        if base_dir is None:
            base_dir = os.path.dirname(__file__)
        self.base_dir = base_dir
        self._templates_dir = os.path.join(base_dir, "templates")
        self._cache: dict[str, str] = {}
        self._warnings: list[str] = []

    def get_template(self, template_path: str) -> str | None:
        """Load a single template by its relative path.

        Accepts paths with or without the bundled `templates/` prefix (e.g.
        both 'templates/ballad-cinematic-pop_0001.txt' and
        'ballad-cinematic-pop_0001.txt' resolve to the same file).

        Returns the template content or None if not found.
        """
        if template_path in self._cache:
            return self._cache[template_path]

        normalized = template_path.replace("\\", "/")
        if normalized.startswith("templates/"):
            normalized = normalized[len("templates/"):]

        full_path = os.path.join(self._templates_dir, normalized)
        if not os.path.isfile(full_path):
            msg = f"Template not found: {template_path}"
            self._warnings.append(msg)
            return None

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        self._cache[template_path] = content
        return content

    def get_templates(self, paths: list[str]) -> dict[str, str]:
        """Load multiple templates. Returns dict of path → content for found templates."""
        result = {}
        for path in paths:
            content = self.get_template(path)
            if content is not None:
                result[path] = content
        return result

    @property
    def warnings(self) -> list[str]:
        """Return any warnings about missing templates."""
        return self._warnings.copy()

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()
        self._warnings.clear()
