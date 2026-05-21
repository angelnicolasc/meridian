"""Telemetry surface for the Meridian plugin.

Prometheus is the primary metric surface — the Rust core emits Prometheus
counters and the plugin mirrors a few Python-side ones here. When OTLP export
is turned on via :func:`init_otlp` (config ``[telemetry] otlp_enabled``), the
same counters are *also* recorded as OpenTelemetry instruments so an OTLP
collector sees them without a separate Prometheus scrape.

The OpenTelemetry instruments are created from the global meter at import time.
Until a meter provider with an exporter is installed they are no-ops, so
importing this module is cheap and has no network effect.
"""

from __future__ import annotations

import logging

from opentelemetry import metrics as _otel_metrics
from prometheus_client import Counter as _PromCounter

logger = logging.getLogger("meridian.telemetry")

# --- Prometheus counters (always live) -------------------------------------

_PROM_BLOCKS_OFFLOADED = _PromCounter(
    "meridian_disagg_blocks_offloaded_total",
    "KV blocks handed to the disaggregation fabric, by fabric label.",
    ["fabric"],
)
_PROM_VOCAB_FALLBACK = _PromCounter(
    "meridian_vocab_fallback_total",
    "Entropy-probe batches that fell back to per-request compute because the "
    "logit rows had heterogeneous vocab sizes.",
)

# --- OpenTelemetry counters (no-op until init_otlp installs a provider) -----

_meter = _otel_metrics.get_meter("meridian")
_OTEL_BLOCKS_OFFLOADED = _meter.create_counter(
    "meridian.disagg.blocks_offloaded",
    description="KV blocks handed to the disaggregation fabric.",
)
_OTEL_VOCAB_FALLBACK = _meter.create_counter(
    "meridian.vocab.fallback",
    description="Entropy-probe per-request fallbacks due to heterogeneous vocab.",
)

_otlp_installed = False


def record_blocks_offloaded(fabric: str, n: int) -> None:
    """Record ``n`` blocks offloaded to ``fabric`` on both metric surfaces."""
    _PROM_BLOCKS_OFFLOADED.labels(fabric=fabric).inc(n)
    _OTEL_BLOCKS_OFFLOADED.add(n, {"fabric": fabric})


def record_vocab_fallback() -> None:
    """Record one heterogeneous-vocab fallback on both metric surfaces."""
    _PROM_VOCAB_FALLBACK.inc()
    _OTEL_VOCAB_FALLBACK.add(1)


def init_otlp(endpoint: str, service_name: str = "meridian") -> None:
    """Install an OTLP/HTTP metric pipeline exporting to ``endpoint``.

    Idempotent: a second call is a no-op. Requires the optional
    ``opentelemetry-exporter-otlp-proto-http`` dependency (``pip install
    'meridian[otel]'``); raises :class:`ImportError` with guidance if it is
    missing.
    """
    global _otlp_installed
    if _otlp_installed:
        return
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError as exc:  # pragma: no cover — exercised only without extra
        msg = (
            "OTLP export requires the 'otel' extra: "
            "pip install 'meridian[otel]'"
        )
        raise ImportError(msg) from exc

    reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
    provider = MeterProvider(
        metric_readers=[reader],
        resource=Resource.create({"service.name": service_name}),
    )
    _otel_metrics.set_meter_provider(provider)
    _otlp_installed = True
    logger.info("OTLP metric export enabled endpoint=%s service=%s", endpoint, service_name)
