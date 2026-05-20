"""Real-world workload loaders for the Meridian benchmark harness.

Loads the two reference datasets the playbook §5 names as the canonical
A/B inputs:

- **ShareGPT** (``anon8231489123/ShareGPT_Vicuna_unfiltered``) — chat
  traffic. Short prompts, short responses, no reasoning. The regime
  where Meridian's win is *protecting* output streaming from
  contention with background reasoning requests.
- **MATH-500** (``HuggingFaceH4/MATH-500``) — competition mathematics.
  Long reasoning chains, the regime where budget forcing and
  dual-queue scheduling actually save GPU-seconds per request.

Both loaders cache to ``~/.cache/meridian/datasets/`` so repeated CI
runs don't re-fetch. The cache is a single Parquet file per dataset;
the harness controls the sample size via ``n_requests`` and a seed.

Network policy
--------------

These loaders are explicitly *opt-in*. The CI default uses
``--workload synthetic`` which never touches the network. Operators
who want a realistic mix pass ``--workload sharegpt`` (or ``math500``)
and the loader pulls from the HuggingFace mirror once, then serves
from the local cache forever.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.workloads import WorkloadRequest

if TYPE_CHECKING:  # pragma: no cover — typing only
    from collections.abc import Iterable

logger = logging.getLogger("meridian.bench.datasets")


def _cache_dir() -> Path:
    """Return the cache directory, creating it on first use."""
    path = Path.home() / ".cache" / "meridian" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_hf_split(repo: str, split: str, cache_name: str) -> list[dict[str, object]]:
    """Pull a HuggingFace dataset split and cache it as a Parquet snapshot.

    The cache path encodes ``repo@split`` so different splits of the same
    repo don't collide. If ``datasets`` isn't installed we raise a clear
    actionable error rather than fall back silently — the user asked for
    a real dataset, so saying "couldn't load it" is the honest answer.
    """
    cache_path = _cache_dir() / f"{cache_name}.parquet"
    try:
        import pyarrow.parquet as pq

        if cache_path.exists():
            table = pq.read_table(cache_path)
            cached: list[dict[str, object]] = table.to_pylist()
            return cached
    except ImportError:
        cache_path = cache_path.with_suffix(".json")
        if cache_path.exists():
            import json

            loaded: list[dict[str, object]] = json.loads(
                cache_path.read_text(encoding="utf-8"),
            )
            return loaded

    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = (
            "real datasets require the `datasets` extra. "
            "Install with `uv pip install datasets pyarrow` or pass "
            "`--workload synthetic`."
        )
        raise SystemExit(msg) from exc

    logger.info("fetching %s split=%s (first run only)", repo, split)
    ds = load_dataset(repo, split=split)
    rows = [dict(r) for r in ds]
    # Persist cache. Parquet is preferred for speed; JSON fallback for
    # environments without pyarrow.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, cache_path)
    except ImportError:
        import json

        cache_path = cache_path.with_suffix(".json")
        cache_path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------


def load_sharegpt(n_requests: int, *, seed: int = 0) -> list[WorkloadRequest]:
    """Load ``n_requests`` single-turn ShareGPT conversations.

    Filters to conversations whose first ``human`` turn is short enough
    to fit a 4 KiB prompt budget — this matches realistic chat traffic
    and avoids occasional ShareGPT outliers that are essentially pasted
    documents.
    """
    rows = _load_hf_split(
        "anon8231489123/ShareGPT_Vicuna_unfiltered", "train", "sharegpt",
    )
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[WorkloadRequest] = []
    for r in rows:
        if len(out) >= n_requests:
            break
        prompt = _first_human_turn(r)
        if prompt is None or len(prompt) > 4096:
            continue
        # Approximate response length from prompt length: full ShareGPT
        # responses average ~1.5x prompt; cap to keep runs deterministic.
        output_tokens = min(512, max(32, len(prompt) // 3))
        out.append(WorkloadRequest(
            request_id=f"sharegpt-{len(out):05d}",
            prompt=prompt,
            kind="chat",
            expected_think_tokens=0,
            expected_output_tokens=output_tokens,
        ))
    return out


def _first_human_turn(row: dict[str, object]) -> str | None:
    conv = row.get("conversations")
    if not isinstance(conv, list):
        return None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role")
        text = turn.get("value") or turn.get("content")
        if role in ("human", "user") and isinstance(text, str) and text.strip():
            return text.strip()
    return None


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------


def load_math500(n_requests: int, *, seed: int = 0) -> list[WorkloadRequest]:
    """Load ``n_requests`` MATH-500 problems as reasoning-style workload.

    Each problem becomes a reasoning request with a generous think budget
    (the reference DeepSeek-R1 solution averages ~3500 think tokens on
    these). Output is the boxed final answer — short.
    """
    rows = _load_hf_split("HuggingFaceH4/MATH-500", "test", "math500")
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[WorkloadRequest] = []
    for r in rows:
        if len(out) >= n_requests:
            break
        problem = r.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            continue
        out.append(WorkloadRequest(
            request_id=f"math500-{len(out):05d}",
            prompt=problem.strip(),
            kind="reasoning",
            expected_think_tokens=3500,
            expected_output_tokens=128,
        ))
    return out


# ---------------------------------------------------------------------------
# Mixed
# ---------------------------------------------------------------------------


def load_mixed(
    n_requests: int,
    reasoning_ratio: float,
    *,
    seed: int = 0,
) -> list[WorkloadRequest]:
    """Blend MATH-500 and ShareGPT at ``reasoning_ratio``.

    Realistic production mix — most inference traffic is conversational,
    a meaningful slice is reasoning. The blend exercises both queues.
    """
    if not 0.0 <= reasoning_ratio <= 1.0:
        msg = f"reasoning_ratio must be in [0, 1], got {reasoning_ratio}"
        raise ValueError(msg)
    n_reasoning = round(n_requests * reasoning_ratio)
    n_chat = n_requests - n_reasoning
    reasoning = load_math500(n_reasoning, seed=seed)
    chat = load_sharegpt(n_chat, seed=seed + 1)
    merged: list[WorkloadRequest] = list(_interleave(reasoning, chat, seed=seed + 2))
    return merged


def _interleave(
    a: Iterable[WorkloadRequest],
    b: Iterable[WorkloadRequest],
    *,
    seed: int,
) -> Iterable[WorkloadRequest]:
    """Deterministic seeded interleave of two iterables."""
    rng = random.Random(seed)
    items = list(a) + list(b)
    rng.shuffle(items)
    return items
