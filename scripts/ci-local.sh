#!/usr/bin/env bash
# Run the full CI matrix locally. Mirrors .github/workflows/ci.yml.
# Fails fast on the first red step.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> [1/8] cargo fmt"
cargo fmt --all -- --check

echo "==> [2/8] cargo clippy"
cargo clippy --workspace --all-targets --all-features -- -D warnings

echo "==> [3/8] cargo nextest (workspace)"
if command -v cargo-nextest >/dev/null 2>&1; then
    cargo nextest run --workspace --all-features
else
    echo "    (cargo-nextest missing; falling back to cargo test)"
    cargo test --workspace --all-features
fi

echo "==> [4/8] cargo doc"
cargo doc --workspace --no-deps --all-features

echo "==> [5/8] cargo deny check"
if command -v cargo-deny >/dev/null 2>&1; then
    cargo deny check
else
    echo "    (cargo-deny missing; skipping)"
fi

echo "==> [6/8] python lint + types"
( cd python && uv sync --extra dev >/dev/null )
( cd python && uv run ruff check meridian tests )
( cd python && uv run mypy --strict meridian )

echo "==> [7/8] pytest (no GPU, no vLLM)"
( cd python && uv run pytest -m "not gpu and not vllm" )

echo "==> [8/8] mdbook build"
if command -v mdbook >/dev/null 2>&1; then
    mdbook build docs
else
    echo "    (mdbook missing; skipping)"
fi

echo
echo "==> All checks passed."
