#!/usr/bin/env python3
"""Keep the version shown on the documentation home page in step with the package.

``src/__init__.py`` holds the single version value for the project;
``pyproject.toml`` and ``conda/meta.yaml`` both read it. The documentation home
page states the version too, and being hand-written it went stale: the site
advertised 1.2.0 while the package built as 1.3.0.

This script renders that line into ``documentation/index.md`` between the
markers below, so the version a reader sees is the version the package ships.

Usage
-----
    python scripts/sync_docs_version.py            # rewrite the version line
    python scripts/sync_docs_version.py --check    # fail if it is stale

Note on dates
-------------
Deliberately no "last updated" date. The generated output is compared against
the committed file by CI, so a value derived from the current date would make
the check fail on the day after every commit. Use the repository history for
dates.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "src" / "__init__.py"
INDEX = ROOT / "documentation" / "index.md"

BEGIN = "<!-- BEGIN GENERATED VERSION -->"
END = "<!-- END GENERATED VERSION -->"

VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)


def package_version(init: Path = INIT) -> str:
    """Return the project version declared in ``src/__init__.py``.

    Parameters
    ----------
    init : Path
        Path to the package ``__init__.py`` holding ``__version__``.

    Returns
    -------
    str
        The version string.

    Raises
    ------
    SystemExit
        If no ``__version__`` assignment is found.
    """
    match = VERSION_RE.search(init.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"No __version__ assignment found in {init}")
    return match.group(1)


def build_block(version: str) -> str:
    """Render the marker-delimited version block."""
    return "\n".join(
        [
            BEGIN,
            "",
            f"**Version**: {version}",
            "",
            END,
        ]
    )


def splice(text: str, block: str) -> str:
    """Replace the marker-delimited block in ``text`` with ``block``."""
    start = text.find(BEGIN)
    end = text.find(END)
    if start == -1 or end == -1:
        raise SystemExit(
            f"Markers not found in {INDEX.name}. Expected {BEGIN} and {END}."
        )
    return text[:start] + block + text[end + len(END) :]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the version sync."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if the version line is out of date",
    )
    args = parser.parse_args(argv)

    version = package_version()
    current = INDEX.read_text(encoding="utf-8")
    updated = splice(current, build_block(version))

    if args.check:
        if current != updated:
            print(
                f"The version shown in {INDEX.relative_to(ROOT)} does not match "
                f"src/__init__.py ({version}).\n"
                "Run: python scripts/sync_docs_version.py",
                file=sys.stderr,
            )
            return 1
        print(f"Documentation version is up to date ({version}).")
        return 0

    INDEX.write_text(updated, encoding="utf-8")
    print(f"Synced version {version} into {INDEX.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
