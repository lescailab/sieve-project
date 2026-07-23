#!/usr/bin/env python3
"""Keep the installed-command table in the command reference in step with ``pyproject.toml``.

``pip install sieve`` installs a console entry point for each name under
``[project.scripts]``. This script renders those names, and the script path
each one resolves to, into a table in ``documentation/command-reference.md``
delimited by the markers below, so the table cannot drift from the packaging
metadata.

Usage
-----
    python scripts/sync_command_table.py            # rewrite the table
    python scripts/sync_command_table.py --check    # fail if it is stale

The ``--check`` mode is what CI runs: it fails when an entry point is missing
from the reference, when one is listed that no longer exists, or when the
rendered table differs in any other way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10, which this project still supports
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as error:  # pragma: no cover - interpreter-dependent
        raise SystemExit(
            "Reading pyproject.toml needs a TOML parser. Python 3.11+ has one "
            "built in; on Python 3.10 install the backport with:\n"
            "    pip install tomli\n"
            "or install the development extra: pip install -e '.[dev]'"
        ) from error

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REFERENCE = ROOT / "documentation" / "command-reference.md"

BEGIN = "<!-- BEGIN GENERATED COMMAND TABLE -->"
END = "<!-- END GENERATED COMMAND TABLE -->"


def load_entry_points(pyproject: Path = PYPROJECT) -> dict[str, str]:
    """Return the ``[project.scripts]`` mapping of command name to target.

    Parameters
    ----------
    pyproject : Path
        Path to ``pyproject.toml``.

    Returns
    -------
    dict of str to str
        Command name mapped to its ``module:function`` target.
    """
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)
    return config["project"]["scripts"]


def script_path(target: str) -> str:
    """Convert a ``scripts.name:main`` entry-point target to its file path."""
    module = target.split(":", 1)[0]
    return module.replace(".", "/") + ".py"


def render_table(entry_points: dict[str, str]) -> str:
    """Render the installed-command table as Markdown."""
    rows = [
        "| Installed command | Script |",
        "|-------------------|--------|",
    ]
    for name in sorted(entry_points):
        rows.append(f"| `{name}` | `{script_path(entry_points[name])}` |")
    return "\n".join(rows)


def build_block(entry_points: dict[str, str]) -> str:
    """Render the full marker-delimited block, table included."""
    return "\n".join(
        [
            BEGIN,
            "",
            f"Installing SIEVE (`pip install sieve`) provides {len(entry_points)} console",
            "commands. Each is equivalent to running the script it points at, so",
            "`sieve-train --help` and `python scripts/train.py --help` are the same",
            "command. The installed form is the one to use from an installed",
            "environment; the script path is for contributors working from a checkout.",
            "",
            "This table is generated from `[project.scripts]` in `pyproject.toml` by",
            "`scripts/sync_command_table.py`. Do not edit it by hand.",
            "",
            render_table(entry_points),
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
            f"Markers not found in {REFERENCE.name}. Expected {BEGIN} and {END}."
        )
    return text[:start] + block + text[end + len(END) :]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the command-table sync."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if the table is out of date",
    )
    args = parser.parse_args(argv)

    entry_points = load_entry_points()
    current = REFERENCE.read_text(encoding="utf-8")
    updated = splice(current, build_block(entry_points))

    if args.check:
        missing = [name for name in entry_points if f"`{name}`" not in current]
        if missing:
            print(
                "Entry points missing from "
                f"{REFERENCE.relative_to(ROOT)}: {', '.join(sorted(missing))}",
                file=sys.stderr,
            )
            return 1
        if current != updated:
            print(
                f"The generated command table in {REFERENCE.relative_to(ROOT)} is "
                "out of date.\nRun: python scripts/sync_command_table.py",
                file=sys.stderr,
            )
            return 1
        print(f"Command table is up to date ({len(entry_points)} entry points).")
        return 0

    REFERENCE.write_text(updated, encoding="utf-8")
    print(
        f"Synced {len(entry_points)} entry points into "
        f"{REFERENCE.relative_to(ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
