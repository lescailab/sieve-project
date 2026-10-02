# About this directory

The Markdown files in `documentation/` are written and edited by hand, and they
are the single source of truth for the SIEVE documentation site published at
<https://lescailab.github.io/sieve-project/>.

The top-level `USER_GUIDE.md` is generated from these pages by
`scripts/assemble_user_guide.py`, which concatenates them in `mkdocs.yml`
navigation order and demotes each page's headings by one level.

Editing `USER_GUIDE.md` directly will fail the `docs-drift` CI job: change the
relevant page here instead, then run `python scripts/assemble_user_guide.py`
and commit both.

Two regions inside these pages are generated too, each between HTML comment
markers, and each checked by the same CI job:

- the Installed Commands table in `command-reference.md`, from
  `[project.scripts]` in `pyproject.toml`, by `scripts/sync_command_table.py`
- the version line in `index.md`, from `__version__` in `src/__init__.py`, by
  `scripts/sync_docs_version.py`

Adding or renaming a console entry point, or bumping the version, without
rerunning the corresponding script fails CI.
