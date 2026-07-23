#!/usr/bin/env python3
"""Keep the version shown on the documentation home page in step with the package.

``src/__init__.py`` holds the single version value for the project;
``pyproject.toml`` and ``conda/meta.yaml`` both read it. The documentation home
page states the version too, and being hand-written it went stale: the site
advertised 1.2.0 while the package built as 1.3.0.

This script renders that line into ``documentation/index.md`` between the
markers below, so the version a reader sees is the version the package ships.

The block also carries the date of the latest commit, so the page states when
the code it documents last changed.

Usage
-----
    python scripts/sync_docs_version.py            # rewrite the block
    python scripts/sync_docs_version.py --check    # fail if the version is stale

How the date stays honest
-------------------------
A date baked into a committed file is stale the moment the next commit lands,
and a strict diff against the committed copy would then fail every build. So
the date is stamped at *build* time: the deploy workflow runs this script
before ``mkdocs build``, and the published page therefore always shows the
commit that is being deployed.

``--check`` consequently compares the version but ignores the date line. That
keeps the version enforced while letting the committed date be whatever it was
when someone last ran the script, since the published value never comes from
there.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "src" / "__init__.py"
INDEX = ROOT / "documentation" / "index.md"

BEGIN = "<!-- BEGIN GENERATED VERSION -->"
END = "<!-- END GENERATED VERSION -->"

VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)
DATE_LINE_RE = re.compile(r"^\*\*Last updated\*\*: \d{4}-\d{2}-\d{2}$", re.MULTILINE)


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


def latest_commit_date(fallback: str | None = None) -> str | None:
    """Return the committer date of HEAD as ``YYYY-MM-DD``.

    Parameters
    ----------
    fallback : str, optional
        Value to return when the date cannot be read, for example when the
        script runs from an exported tarball rather than a git checkout.

    Returns
    -------
    str or None
        The date, or *fallback* when git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    return result.stdout.strip() or fallback


def existing_date(text: str) -> str | None:
    """Return the date already present in the block, if any."""
    match = DATE_LINE_RE.search(text)
    return match.group(0).rsplit(": ", 1)[1] if match else None


def build_block(version: str, date: str | None) -> str:
    """Render the marker-delimited version block."""
    lines = [BEGIN, "", f"**Version**: {version}"]
    if date:
        lines += ["", f"**Last updated**: {date}"]
    lines += ["", END]
    return "\n".join(lines)


def _ignoring_date(text: str) -> str:
    """Blank the date line so two renderings can be compared without it."""
    return DATE_LINE_RE.sub("**Last updated**: DATE", text)


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
    date = latest_commit_date(fallback=existing_date(current))
    updated = splice(current, build_block(version, date))

    if args.check:
        # The date is stamped at build time, so only the version is enforced.
        if _ignoring_date(current) != _ignoring_date(updated):
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
    print(
        f"Synced version {version} (last commit {date}) into "
        f"{INDEX.relative_to(ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
