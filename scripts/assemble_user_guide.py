#!/usr/bin/env python3
"""Assemble ``USER_GUIDE.md`` from the hand-authored ``documentation/`` pages.

The MkDocs source tree under ``documentation/`` is the single source of truth
for SIEVE documentation. ``USER_GUIDE.md`` is a generated artefact: a
single-file concatenation of those pages, in ``mkdocs.yml`` navigation order,
provided for readers who want the whole guide offline or in one scroll.

Every heading in each page is demoted by one level so that the assembled
document carries exactly one top-level title.

Usage
-----
    python scripts/assemble_user_guide.py            # rewrite USER_GUIDE.md
    python scripts/assemble_user_guide.py --check    # fail if it is stale

The ``--check`` mode is what CI runs: it regenerates the guide in memory and
exits non-zero if the committed file differs, which makes documentation drift
a build failure instead of a silent overwrite.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parent.parent
MKDOCS_CONFIG = ROOT / "mkdocs.yml"
OUTPUT = ROOT / "USER_GUIDE.md"

GUIDE_TITLE = "SIEVE User Guide"

BANNER = f"""<!--
This file is GENERATED. Do not edit it by hand.

Source of truth: the documentation/ directory (rendered at
https://lescailab.github.io/sieve-project/).

Edit the relevant page under documentation/, then run:

    python scripts/assemble_user_guide.py

Direct edits to this file are detected by the docs-drift CI job and will
fail the build.
-->
"""

HEADING_RE = re.compile(r"^(#{1,6})(\s+)(.*)$")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")


class _TolerantLoader(yaml.SafeLoader):
    """A SafeLoader that ignores the Python-object tags MkDocs uses.

    ``mkdocs.yml`` carries ``!!python/name:`` tags for the emoji extension.
    Those are irrelevant to navigation order, so they are resolved to ``None``
    rather than raising.
    """


def _ignore_unknown(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> None:
    return None


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _ignore_unknown)
_TolerantLoader.add_multi_constructor("!", _ignore_unknown)


def load_nav_pages(config_path: Path = MKDOCS_CONFIG) -> list[Path]:
    """Return the documentation pages in ``mkdocs.yml`` navigation order.

    Parameters
    ----------
    config_path : Path
        Path to ``mkdocs.yml``.

    Returns
    -------
    list of Path
        Absolute paths to the Markdown sources, in navigation order.
    """
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=_TolerantLoader)

    docs_dir = ROOT / config.get("docs_dir", "docs")
    return [docs_dir / rel for rel in _walk_nav(config["nav"])]


def _walk_nav(node: Any) -> Iterator[str]:
    """Yield every Markdown path referenced by a (possibly nested) nav node."""
    if isinstance(node, str):
        if node.endswith(".md"):
            yield node
    elif isinstance(node, list):
        for item in node:
            yield from _walk_nav(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk_nav(value)


def demote_headings(text: str) -> str:
    """Demote every Markdown heading in ``text`` by one level.

    Headings inside fenced code blocks are left alone, so that shell comments
    such as ``# 1. Install`` survive intact. Level-6 headings cannot be demoted
    further and are returned unchanged.

    Parameters
    ----------
    text : str
        Markdown source of a single page.

    Returns
    -------
    str
        The same source with headings shifted down one level.
    """
    lines = text.split("\n")
    out: list[str] = []
    fence: str | None = None

    for line in lines:
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(2)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
            out.append(line)
            continue

        if fence is None:
            heading = HEADING_RE.match(line)
            if heading and len(heading.group(1)) < 6:
                line = f"#{heading.group(1)}{heading.group(2)}{heading.group(3)}"

        out.append(line)

    return "\n".join(out)


def _first_heading(text: str) -> str | None:
    """Return the text of the first level-1 heading, if the page opens with one."""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        heading = HEADING_RE.match(stripped)
        if heading and len(heading.group(1)) == 1:
            return heading.group(3).strip()
        return None
    return None


def _strip_first_heading(text: str) -> str:
    """Remove the leading level-1 heading line from ``text``."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.strip():
            return "\n".join(lines[index + 1 :]).lstrip("\n")
    return text


def _slugify(heading: str) -> str:
    """Approximate the anchor MkDocs and GitHub derive from a heading."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[-\s]+", "-", slug).strip("-")


def build_toc(pages: list[Path]) -> str:
    """Build a table of contents linking to each page's top-level heading."""
    entries = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        title = _first_heading(text)
        if title is None or title == GUIDE_TITLE:
            continue
        entries.append(f"- [{title}](#{_slugify(title)})")
    return "## Table of Contents\n\n" + "\n".join(entries) + "\n"


def assemble(pages: list[Path] | None = None) -> str:
    """Assemble the full user guide from the documentation pages.

    Parameters
    ----------
    pages : list of Path, optional
        Pages to concatenate. Defaults to the ``mkdocs.yml`` navigation order.

    Returns
    -------
    str
        The complete contents of ``USER_GUIDE.md``.
    """
    if pages is None:
        pages = load_nav_pages()

    parts = [BANNER, f"# {GUIDE_TITLE}\n", build_toc(pages)]

    for page in pages:
        text = page.read_text(encoding="utf-8")
        if _first_heading(text) == GUIDE_TITLE:
            text = _strip_first_heading(text)
        else:
            text = demote_headings(text)
        parts.append(text.strip("\n") + "\n")

    return "\n".join(parts).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the assembler."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if USER_GUIDE.md is out of date",
    )
    args = parser.parse_args(argv)

    content = assemble()

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != content:
            print(
                f"{OUTPUT.name} is out of date with documentation/.\n"
                "Run: python scripts/assemble_user_guide.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.name} is up to date.")
        return 0

    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} from documentation/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
