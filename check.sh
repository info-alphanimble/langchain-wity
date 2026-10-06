#!/bin/sh
# Everything that must pass before a commit: format, lint, types, tests, build.
set -e
cd "$(dirname "$0")"
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
uv build -q
echo "All checks passed."
