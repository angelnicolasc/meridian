"""Per-request and aggregate metric records emitted by the benchmark harness.

Naming conventions match the playbook §5 metric table so report consumers
can cross-reference directly with the operator-facing Prometheus catalog.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RequestResult:
    """Metrics captured for one served request."""

    req_id: str
    kind: str  # "chat" | "reasoning"

    ttft_ms: float = 0.0
    """Time-to-first-token. Includes prefill + the first decoded token."""

    think_tokens: int = 0
    output_tokens: int = 0

    ttot_ms: float = 0.0
    """Time-to-first-OUTPUT-token: from ``</think>`` emission to the first
    user-visible token after it. The user-visible streaming-start latency."""

    output_itl_ms: list[float] = field(default_factory=list)
    """Inter-token latencies of the output phase. P50 / P95 over this list
    is the streaming-quality KPI."""

    budget_forced: bool = False
    force_reason: str | None = None

    think_converged_at: int = -1
    """Token count when EAT converged (filled by the plugin when available)."""

    output_critical_evictions_during: int = 0
    """How many ``OutputCritical`` evictions fired while this request was
    in output phase. Anything > 0 is a user-visible degradation event."""


def _delta_pct(stock: float, meridian: float) -> float:
    """Signed percent change ``(meridian - stock) / stock * 100``.

    Negative is "Meridian faster / fewer". Returns ``nan`` if the stock
    baseline is non-positive (avoids divide-by-zero noise in reports).
    """
    if not math.isfinite(stock) or stock <= 0 or not math.isfinite(meridian):
        return float("nan")
    return (meridian - stock) / stock * 100.0


def _semaphore(delta_pct: float, *, lower_is_better: bool = True) -> str:
    """Return a coloured-text indicator suitable for a Markdown report.

    The indicator is a plain ASCII token rather than an emoji so the
    output survives terminals and CI log viewers that strip non-ASCII.
    """
    if not math.isfinite(delta_pct):
        return "n/a"
    improved = (delta_pct < 0) if lower_is_better else (delta_pct > 0)
    magnitude = abs(delta_pct)
    if magnitude < 1.0:
        return "FLAT"
    if improved:
        return "WIN" if magnitude >= 5.0 else "win"
    return "LOSS" if magnitude >= 5.0 else "loss"


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregated report over a benchmark run."""

    config_name: str
    duration_s: float
    arrival_rate_rps: float
    n_requests: int
    n_reasoning: int
    n_chat: int

    ttft_p50_ms: float
    ttft_p95_ms: float
    ttot_p50_ms: float
    ttot_p95_ms: float
    output_itl_p50_ms: float
    output_itl_p95_ms: float
    output_itl_p99_ms: float

    think_tokens_avg: float
    think_tokens_p95: float
    output_tokens_avg: float

    budget_forced_pct: float
    budget_forced_by_reason: dict[str, int]
    output_critical_eviction_events: int

    def to_json(self) -> str:
        """Serialise to indent-4 JSON for diffable artefacts."""
        return json.dumps(asdict(self), indent=4, sort_keys=True)

    def to_markdown(self) -> str:
        """Human-readable summary block. Suitable for PR comments."""
        lines = [
            f"# Meridian benchmark: `{self.config_name}`",
            "",
            f"- Duration: **{self.duration_s:.1f} s**, rate **{self.arrival_rate_rps:.1f} req/s**",
            f"- Requests: {self.n_requests} "
            f"(reasoning {self.n_reasoning}, chat {self.n_chat})",
            "",
            "## Latency",
            "",
            "| Metric                       | P50      | P95      | P99      |",
            "|------------------------------|----------|----------|----------|",
            (
                f"| TTFT (prefill + 1st token)   "
                f"| {self.ttft_p50_ms:7.1f}  | {self.ttft_p95_ms:7.1f}  |   --     |"
            ),
            (
                f"| TTOT (think -> 1st output)   "
                f"| {self.ttot_p50_ms:7.1f}  | {self.ttot_p95_ms:7.1f}  |   --     |"
            ),
            (
                f"| Output ITL (stream jitter)   | {self.output_itl_p50_ms:7.2f}  "
                f"| {self.output_itl_p95_ms:7.2f}  | {self.output_itl_p99_ms:7.2f}  |"
            ),
            "",
            "## Reasoning",
            "",
            f"- Think tokens — avg **{self.think_tokens_avg:.0f}**, "
            f"P95 **{self.think_tokens_p95:.0f}**",
            f"- Output tokens — avg **{self.output_tokens_avg:.0f}**",
            f"- Budget forced: **{self.budget_forced_pct:.1f}%** of reasoning requests",
            f"- Force reason breakdown: {self.budget_forced_by_reason}",
            "",
            "## KV pressure",
            "",
            f"- `OutputCritical` eviction events: **{self.output_critical_eviction_events}** "
            "(any non-zero is alertable)",
        ]
        return "\n".join(lines)

    def write_artifacts(self, out_dir: str | Path) -> None:
        """Write both JSON and Markdown reports under ``out_dir``."""
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.json").write_text(self.to_json(), encoding="utf-8")
        (path / "report.md").write_text(self.to_markdown(), encoding="utf-8")


# Metrics shown in the A/B table. Each is (label, BenchmarkReport attr,
# lower_is_better). All current metrics are lower-is-better.
_AB_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("TTFT P50 (ms)",            "ttft_p50_ms",                    True),
    ("TTFT P95 (ms)",            "ttft_p95_ms",                    True),
    ("TTOT P50 (ms)",            "ttot_p50_ms",                    True),
    ("TTOT P95 (ms)",            "ttot_p95_ms",                    True),
    ("Output ITL P50",          "output_itl_p50_ms",              True),
    ("Output ITL P95",          "output_itl_p95_ms",              True),
    ("Output ITL P99",          "output_itl_p99_ms",              True),
    ("Think tokens avg",        "think_tokens_avg",               True),
    ("OutputCritical evictions", "output_critical_eviction_events", True),
)


@dataclass(slots=True)
class ABComparisonReport:
    """A/B comparison of N scheduler runs against one run under test.

    ``reports`` maps a config name to its :class:`BenchmarkReport`; the run
    keyed by ``under_test`` (Meridian) is the one being evaluated, and a delta
    column is produced for it against every other (baseline) report. The
    common case is a single baseline (stock or static-budget) versus Meridian;
    additional baselines simply add columns.
    """

    reports: dict[str, BenchmarkReport]
    under_test: str

    def __post_init__(self) -> None:
        if self.under_test not in self.reports:
            msg = f"under_test {self.under_test!r} not among reports {list(self.reports)}"
            raise KeyError(msg)

    @classmethod
    def pair(cls, baseline: BenchmarkReport, meridian: BenchmarkReport) -> ABComparisonReport:
        """Convenience constructor for the single-baseline case."""
        return cls(
            reports={baseline.config_name: baseline, meridian.config_name: meridian},
            under_test=meridian.config_name,
        )

    def _baseline_names(self) -> list[str]:
        return [name for name in self.reports if name != self.under_test]

    def to_json(self) -> str:
        """Serialise every report plus the under-test deltas against each baseline."""
        ut = self.reports[self.under_test]
        deltas = {
            base: {
                attr: _delta_pct(
                    float(getattr(self.reports[base], attr)),
                    float(getattr(ut, attr)),
                )
                for _label, attr, _lb in _AB_METRICS
            }
            for base in self._baseline_names()
        }
        payload = {
            "under_test": self.under_test,
            "reports": {name: asdict(rep) for name, rep in self.reports.items()},
            "delta_pct": deltas,
        }
        return json.dumps(payload, indent=4, sort_keys=True)

    def to_markdown(self) -> str:
        """Render an N-way value table plus a Δ(%) column per baseline.

        Lower-is-better metrics flag ``WIN`` / ``win`` / ``FLAT`` / ``loss``
        / ``LOSS`` relative to each baseline. The deltas are also in the JSON
        form so dashboards can colour-code without re-parsing.
        """
        ut = self.reports[self.under_test]
        baselines = self._baseline_names()
        value_cols = [*baselines, self.under_test]

        header = ["Metric", *value_cols, *(f"Δ% vs {b}" for b in baselines)]
        sep = ["---"] * len(header)
        lines = [
            f"# A/B comparison: under test `{self.under_test}`",
            "",
            "- Runs: "
            + ", ".join(
                f"`{name}` ({rep.n_requests} req, reasoning {rep.n_reasoning})"
                for name, rep in self.reports.items()
            ),
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for label, attr, lower_better in _AB_METRICS:
            values = [float(getattr(self.reports[name], attr)) for name in value_cols]
            cells = [f"{v:.2f}" for v in values]
            delta_cells: list[str] = []
            ut_val = float(getattr(ut, attr))
            for base in baselines:
                delta = _delta_pct(float(getattr(self.reports[base], attr)), ut_val)
                sem = _semaphore(delta, lower_is_better=lower_better)
                delta_cells.append("n/a" if not math.isfinite(delta) else f"{delta:+.2f} {sem}")
            lines.append("| " + " | ".join([label, *cells, *delta_cells]) + " |")

        lines.extend([
            "",
            "## Budget forcing",
            "",
            *(
                f"- `{name}`: **{rep.budget_forced_pct:.1f}%** of reasoning requests forced "
                f"({rep.budget_forced_by_reason})"
                for name, rep in self.reports.items()
            ),
        ])
        return "\n".join(lines)

    def write_artifacts(self, out_dir: str | Path) -> None:
        """Write ``ab-report.json`` and ``ab-report.md`` under ``out_dir``."""
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "ab-report.json").write_text(self.to_json(), encoding="utf-8")
        (path / "ab-report.md").write_text(self.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _safe_percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted(xs)[int(k)]
    s = sorted(xs)
    return s[f] + (s[c] - s[f]) * (k - f)


def aggregate(
    results: list[RequestResult],
    *,
    config_name: str,
    duration_s: float,
    arrival_rate_rps: float,
    output_critical_eviction_events: int,
) -> BenchmarkReport:
    """Build a :class:`BenchmarkReport` from raw per-request results."""
    if not results:
        msg = "cannot aggregate over an empty result set"
        raise ValueError(msg)

    n_reasoning = sum(1 for r in results if r.kind == "reasoning")
    n_chat = sum(1 for r in results if r.kind == "chat")

    ttfts = [r.ttft_ms for r in results if r.ttft_ms > 0]
    ttots = [r.ttot_ms for r in results if r.ttot_ms > 0]
    itls = [x for r in results for x in r.output_itl_ms]
    think_tokens = [r.think_tokens for r in results if r.kind == "reasoning"]
    output_tokens = [r.output_tokens for r in results]

    forced = [r for r in results if r.budget_forced]
    by_reason: dict[str, int] = {}
    for r in forced:
        key = r.force_reason or "unknown"
        by_reason[key] = by_reason.get(key, 0) + 1

    forced_pct = (
        100.0 * len(forced) / max(1, n_reasoning) if n_reasoning > 0 else 0.0
    )

    return BenchmarkReport(
        config_name=config_name,
        duration_s=duration_s,
        arrival_rate_rps=arrival_rate_rps,
        n_requests=len(results),
        n_reasoning=n_reasoning,
        n_chat=n_chat,
        ttft_p50_ms=_safe_percentile(ttfts, 50),
        ttft_p95_ms=_safe_percentile(ttfts, 95),
        ttot_p50_ms=_safe_percentile(ttots, 50),
        ttot_p95_ms=_safe_percentile(ttots, 95),
        output_itl_p50_ms=_safe_percentile(itls, 50),
        output_itl_p95_ms=_safe_percentile(itls, 95),
        output_itl_p99_ms=_safe_percentile(itls, 99),
        think_tokens_avg=statistics.fmean(think_tokens) if think_tokens else 0.0,
        think_tokens_p95=_safe_percentile([float(t) for t in think_tokens], 95)
        if think_tokens
        else 0.0,
        output_tokens_avg=statistics.fmean(output_tokens) if output_tokens else 0.0,
        budget_forced_pct=forced_pct,
        budget_forced_by_reason=by_reason,
        output_critical_eviction_events=output_critical_eviction_events,
    )
