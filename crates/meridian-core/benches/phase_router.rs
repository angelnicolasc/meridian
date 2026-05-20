#![allow(missing_docs)]
//! Criterion benchmark for the PhaseRouter hot path.
//!
//! The router is on the critical path of every decode step; budget for the
//! whole `on_token` call is single-digit microseconds. This bench measures the
//! steady-state cost of processing a normal think token in `ThinkDecode`.
//!
//! Run with: `cargo bench -p meridian-core --bench phase_router`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use meridian_core::phase_router::{PhaseRouter, PhaseRouterConfig};
use meridian_core::types::EntropySignal;

const THINK_START: u32 = 1;
const THINK_END: u32 = 2;
const EOS: u32 = 3;
const NORMAL: u32 = 100;

fn router() -> PhaseRouter {
    PhaseRouter::new(PhaseRouterConfig {
        max_think_tokens: 1_000_000,
        min_think_tokens: 8,
        think_start_ids: vec![THINK_START],
        think_end_ids: vec![THINK_END],
        eos_ids: vec![EOS],
        ..PhaseRouterConfig::default()
    })
}

fn bench_on_token_no_signal(c: &mut Criterion) {
    let r = router();
    r.register(0);
    r.on_token(0, THINK_START, None);

    c.bench_function("phase_router::on_token (think, no signal)", |b| {
        b.iter(|| {
            black_box(r.on_token(black_box(0), black_box(NORMAL), None));
        });
    });
}

fn bench_on_token_with_signal(c: &mut Criterion) {
    let r = router();
    r.register(0);
    r.on_token(0, THINK_START, None);

    let sig = EntropySignal {
        token_entropy: 1.5,
        eat: 0.2,
        eat_ema: 0.2,
        eat_ema_variance: 0.5,
    };

    c.bench_function("phase_router::on_token (think, entropy signal)", |b| {
        b.iter(|| {
            black_box(r.on_token(black_box(0), black_box(NORMAL), Some(&sig)));
        });
    });
}

criterion_group!(
    benches,
    bench_on_token_no_signal,
    bench_on_token_with_signal
);
criterion_main!(benches);
