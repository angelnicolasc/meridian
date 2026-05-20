"""Workload definitions for the Meridian benchmark harness.

Two reference workloads are supplied:

- **ShareGPT mix**: short prompts, mostly non-reasoning responses. Represents
  conversational traffic where Meridian's value is *protecting* output TTOT
  from background think-heavy requests.
- **MATH-500**: math benchmark prompts. Long reasoning chains, the regime
  where budget forcing and dual-queue scheduling actually save GPU time.

The harness blends them at a configurable ratio. For CI without GPU access,
the synthetic generator below produces deterministic placeholder prompts
that exercise the scheduler decision surface without needing a real model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class WorkloadRequest:
    """A single workload request."""

    request_id: str
    """Stable id used to tag every metric this request emits."""

    prompt: str
    """The actual prompt string sent to the model."""

    kind: Literal["chat", "reasoning"]
    """``"chat"`` for ShareGPT-style, ``"reasoning"`` for MATH-style."""

    expected_think_tokens: int
    """Synthetic expectation: how many think tokens this prompt would produce
    if served by a reasoning model. Used only by the synthetic-replay mode."""

    expected_output_tokens: int
    """Synthetic expectation: post-`</think>` output token count."""


# ---------------------------------------------------------------------------
# Synthetic generators — used by CI without GPU
# ---------------------------------------------------------------------------


_CHAT_TEMPLATES: tuple[str, ...] = (
    "Summarise the following article in two sentences: ...",
    "Translate this paragraph to Spanish: ...",
    "Rewrite the following bullet list as a coherent paragraph: ...",
    "Suggest three alternative subject lines for this email: ...",
    "Explain this concept to a five-year-old: ...",
)

_REASONING_TEMPLATES: tuple[str, ...] = (
    "Solve the integral of x^2 sin(x) dx step by step.",
    "Prove that the sum of the first n odd numbers equals n squared.",
    "Find all real solutions of x^4 - 5x^2 + 4 = 0 and justify each step.",
    "Three boxes contain coins; one contains exactly twice as many as another. ...",
    "An ant starts at (0,0) and moves one unit per second. Compute ...",
)


def synthetic_chat_request(req_id: str, rng: random.Random) -> WorkloadRequest:
    """A short chat-style request with low expected think footprint."""
    return WorkloadRequest(
        request_id=req_id,
        prompt=rng.choice(_CHAT_TEMPLATES),
        kind="chat",
        expected_think_tokens=0,
        expected_output_tokens=rng.randint(40, 240),
    )


def synthetic_reasoning_request(req_id: str, rng: random.Random) -> WorkloadRequest:
    """A reasoning-style request with a heavy think footprint."""
    return WorkloadRequest(
        request_id=req_id,
        prompt=rng.choice(_REASONING_TEMPLATES),
        kind="reasoning",
        # MATH-500 reasoning chains land between ~600 and ~6k tokens on
        # DeepSeek-R1-Distill-Llama-70B; pick a wide range so the budget
        # forcing heuristics see realistic distributions.
        expected_think_tokens=rng.randint(600, 6_000),
        expected_output_tokens=rng.randint(60, 400),
    )


def synthetic_mix(
    n_requests: int,
    reasoning_ratio: float,
    *,
    seed: int = 0xC0FFEE,
) -> list[WorkloadRequest]:
    """Generate ``n_requests`` with the given reasoning ratio.

    The output is deterministic given ``seed`` — the harness uses this so
    CI runs produce comparable numbers across PRs.
    """
    if not 0.0 <= reasoning_ratio <= 1.0:
        msg = "reasoning_ratio must lie in [0, 1]"
        raise ValueError(msg)

    rng = random.Random(seed)
    out: list[WorkloadRequest] = []
    for i in range(n_requests):
        rid = f"syn-{i:06d}"
        if rng.random() < reasoning_ratio:
            out.append(synthetic_reasoning_request(rid, rng))
        else:
            out.append(synthetic_chat_request(rid, rng))
    return out
