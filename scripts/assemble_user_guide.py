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
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


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
    such as ``# 1. Install`` survive intact. Fence tracking follows CommonMark:
    a block opened with N or more backticks (or tildes) is closed only by a
    run of the same character that is at least as long and carries no info
    string, so a page may nest a three-backtick example inside a four-backtick
    fence. Level-6 headings cannot be demoted further and are returned
    unchanged.

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
            marker, info = fence_match.group(2), fence_match.group(3)
            if fence is None:
                fence = marker
            elif (
                marker[0] == fence[0]
                and len(marker) >= len(fence)
                and not info.strip()
            ):
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


LINK_RE = re.compile(r"\]\(([^)\s]+\.md)(#[^)\s]*)?\)")


def page_anchors(pages: list[Path]) -> dict[str, str]:
    """Map each page's path, as pages link to it, to its in-document anchor.

    A cross-page link such as ``](command-reference.md)`` is correct on the
    MkDocs site, where each page is its own URL, but dangles in the assembled
    single-file guide. This mapping lets those links be rewritten to the
    anchor of the corresponding heading.

    Parameters
    ----------
    pages : list of Path
        The documentation pages being assembled.

    Returns
    -------
    dict of str to str
        Both the bare filename and the path relative to ``documentation/``,
        mapped to the anchor derived from that page's level-1 heading.
    """
    docs_dir = pages[0].parent if pages else Path()
    while docs_dir.name == "appendices":
        docs_dir = docs_dir.parent

    anchors: dict[str, str] = {}
    for page in pages:
        title = _first_heading(page.read_text(encoding="utf-8"))
        anchor = "#" + _slugify(title) if title else ""
        if not anchor:
            continue
        anchors[page.name] = anchor
        try:
            anchors[page.relative_to(docs_dir).as_posix()] = anchor
        except ValueError:
            pass
    return anchors


def rewrite_internal_links(text: str, anchors: dict[str, str]) -> str:
    """Rewrite cross-page Markdown links to in-document anchors.

    ``](page.md)`` becomes ``](#page-title)`` and ``](page.md#section)``
    becomes ``](#section)``, since a heading keeps its anchor when demoted.
    Links to targets outside the assembled set are left alone.

    Parameters
    ----------
    text : str
        Markdown source of a single page.
    anchors : dict of str to str
        Mapping from page path to in-document anchor, from `page_anchors`.

    Returns
    -------
    str
        The source with cross-page links pointing inside the assembled guide.
    """

    def replace(match: re.Match[str]) -> str:
        target, fragment = match.group(1), match.group(2)
        key = target.lstrip("./")
        if key not in anchors:
            return match.group(0)
        return f"]({fragment})" if fragment else f"]({anchors[key]})"

    return LINK_RE.sub(replace, text)


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
    anchors = page_anchors(pages)

    for page in pages:
        text = rewrite_internal_links(page.read_text(encoding="utf-8"), anchors)
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
