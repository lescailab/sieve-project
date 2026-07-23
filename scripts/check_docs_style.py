#!/usr/bin/env python3
"""Enforce the project's prose style rules across documentation and source comments.

Two rules are checked:

1. **No em dashes.** Use hyphens, commas, colons or parentheses instead.
2. **British English.** American spellings of the ``-ize``/``-ise`` and
   ``-or``/``-our`` families are rejected in prose.

Identifiers, keyword arguments and library-imposed spellings are out of scope:
only Markdown prose, Python comments and Python docstrings are inspected. In
Python files the checker skips code lines entirely, so a call to
``sklearn``'s ``normalize`` or a variable named ``optimizer`` does not trip it.

Usage
-----
    python scripts/check_docs_style.py            # check the tracked tree
    python scripts/check_docs_style.py FILE...    # check specific files

Exits non-zero and prints ``path:line: message`` for each violation.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EM_DASH = "—"

# American spellings rejected in prose, mapped to the British form to suggest.
AMERICAN_SPELLINGS = {
    "modeling": "modelling",
    "modeled": "modelled",
    "regularize": "regularise",
    "regularized": "regularised",
    "regularizing": "regularising",
    "regularization": "regularisation",
    "visualize": "visualise",
    "visualized": "visualised",
    "visualizing": "visualising",
    "visualization": "visualisation",
    "optimize": "optimise",
    "optimized": "optimised",
    "optimizing": "optimising",
    "optimization": "optimisation",
    "summarize": "summarise",
    "summarized": "summarised",
    "summarizing": "summarising",
    "normalize": "normalise",
    "normalized": "normalised",
    "normalizing": "normalising",
    "normalization": "normalisation",
    "analyze": "analyse",
    "analyzed": "analysed",
    "analyzing": "analysing",
    "behavior": "behaviour",
    "behaviors": "behaviours",
    "behavioral": "behavioural",
    "prioritize": "prioritise",
    "prioritized": "prioritised",
    "binarize": "binarise",
    "binarized": "binarised",
    "penalize": "penalise",
    "penalized": "penalised",
}

SPELLING_RE = re.compile(
    r"\b(" + "|".join(sorted(AMERICAN_SPELLINGS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Paths never checked: generated files, vendored trees, build output.
EXCLUDED = {
    "USER_GUIDE.md",  # generated from documentation/
}
EXCLUDED_PREFIXES = ("site/", ".venv/", "conda-build/", "utilities/demos/")

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"``[^`]+``|`[^`]*`")
MD_LINK_TARGET_RE = re.compile(r"\]\([^)]*\)")


def tracked_files() -> list[Path]:
    """Return the tracked Markdown and Python files in scope."""
    out = subprocess.run(
        ["git", "ls-files", "*.md", "src/*.py", "src/**/*.py", "scripts/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    paths = []
    for rel in out:
        if rel in EXCLUDED or rel.startswith(EXCLUDED_PREFIXES):
            continue
        paths.append(ROOT / rel)
    return paths


def _strip_code_spans(line: str) -> str:
    """Blank out inline code spans and link targets so identifiers are ignored.

    Covers both Markdown ``` `code` ``` and the RST ``` ``code`` ``` form used
    in this project's docstrings, so that naming a library function such as
    ``normalize`` in prose does not trip the spelling rule.
    """
    line = INLINE_CODE_RE.sub("`code`", line)
    return MD_LINK_TARGET_RE.sub("]()", line)


def _match_case(source: str, replacement: str) -> str:
    """Return ``replacement`` cased to match ``source``."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def prose_lines(path: Path) -> list[tuple[int, str]]:
    """Yield the (1-indexed line number, text) pairs that count as prose.

    For Markdown, everything outside fenced code blocks, with inline code spans
    and link targets blanked. For Python, comments and string literals that look
    like docstrings, since those are the only places prose lives.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    result: list[tuple[int, str]] = []

    if path.suffix == ".md":
        fence: str | None = None
        for number, line in enumerate(lines, start=1):
            fence_match = FENCE_RE.match(line)
            if fence_match:
                marker, info = fence_match.group(2), fence_match.group(3)
                if fence is None:
                    fence = marker
                elif marker[0] == fence[0] and len(marker) >= len(fence) and not info.strip():
                    fence = None
                continue
            if fence is None:
                result.append((number, _strip_code_spans(line)))
        return result

    in_docstring: str | None = None
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if in_docstring:
            result.append((number, _strip_code_spans(line)))
            if in_docstring in line:
                in_docstring = None
            continue
        for quote in ('"""', "'''"):
            if stripped.startswith(quote) or f"= {quote}" in stripped:
                result.append((number, _strip_code_spans(line)))
                if stripped.count(quote) == 1:
                    in_docstring = quote
                break
        else:
            if "#" in line:
                result.append((number, _strip_code_spans(line[line.index("#") :])))
    return result


def check_file(path: Path) -> list[str]:
    """Return a list of ``path:line: message`` violations for one file."""
    problems = []
    try:
        rel: Path | str = path.relative_to(ROOT)
    except ValueError:  # a path passed explicitly from outside the repository
        rel = path
    for number, line in prose_lines(path):
        if EM_DASH in line:
            problems.append(
                f"{rel}:{number}: em dash. Use a hyphen, comma, colon or parentheses."
            )
        for match in SPELLING_RE.finditer(line):
            word = match.group(1)
            suggestion = _match_case(word, AMERICAN_SPELLINGS[word.lower()])
            problems.append(
                f"{rel}:{number}: American spelling '{word}'. Use '{suggestion}'."
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Entry point for the style check."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="*", type=Path, help="files to check")
    args = parser.parse_args(argv)

    paths = [p.resolve() for p in args.files] if args.files else tracked_files()

    problems: list[str] = []
    for path in paths:
        if path.exists():
            problems.extend(check_file(path))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(
            f"\n{len(problems)} style violation(s). "
            "The rules are British English and no em dashes in prose.",
            file=sys.stderr,
        )
        return 1

    print(f"Style check passed over {len(paths)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
