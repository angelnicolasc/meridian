#!/usr/bin/env bash
# Runs once after the devcontainer is created.
# Goal: leave the workspace ready for the first `cargo nextest run`.

set -euo pipefail

echo "==> Verifying toolchains"
rustc --version
cargo --version
python --version
uv --version

echo "==> Installing pre-commit hooks (if pre-commit available)"
if command -v pre-commit >/dev/null 2>&1; then
    pre-commit install --install-hooks || true
fi

echo "==> Bootstrap cargo build (no-op if cached)"
cargo build --workspace --keep-going

echo "==> Bootstrap uv sync"
( cd python && uv sync --extra dev )

cat <<'EOM'

==============================================================================
Meridian devcontainer ready.

  Rust tests:    cargo nextest run -p meridian-core
  Python tests:  (cd python && uv run pytest -m "not gpu")
  Docs:          mdbook serve docs

EOM
