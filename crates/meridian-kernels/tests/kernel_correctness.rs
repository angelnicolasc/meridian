//! CUDA kernel numeric correctness vs the CPU reference.
//!
//! These tests only run under `cargo test --features cuda` because they
//! require `nvcc` at build time and a CUDA-capable device at runtime. The
//! workspace default builds against the stub which makes `cuda_available()`
//! return `false`; in that case the tests skip with a message.
//!
//! The "CPU reference" is a `numpy`-equivalent implementation in plain Rust
//! using `f64` internally — chosen to match exactly the algorithm the Python
//! reference backend uses (`python/meridian/_backends/cpu.py`).

#![cfg(feature = "cuda")]

use core::ffi::c_void;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;

use meridian_kernels::ffi::Dtype;
use meridian_kernels::{cuda_available, launch_eat, launch_entropy};

// ---------------------------------------------------------------------------
// CPU reference: same algorithm as python/meridian/_backends/cpu.py
// ---------------------------------------------------------------------------

fn cpu_entropy(logits: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &x in logits {
        if x > max {
            max = x;
        }
    }
    let mut z = 0.0_f64;
    let mut weighted = 0.0_f64;
    for &x in logits {
        let shifted = (x - max) as f64;
        let ex = shifted.exp();
        z += ex;
        weighted += ex * shifted;
    }
    (z.ln() - weighted / z) as f32
}

fn cpu_eat(logits: &[f32], end_ids: &[i32]) -> f32 {
    if end_ids.is_empty() {
        return 0.0;
    }
    let mut max = f32::NEG_INFINITY;
    for &x in logits {
        if x > max {
            max = x;
        }
    }
    let mut z = 0.0_f64;
    let mut gathered = 0.0_f64;
    for &x in logits {
        z += ((x - max) as f64).exp();
    }
    for &idx in end_ids {
        if idx >= 0 && (idx as usize) < logits.len() {
            gathered += ((logits[idx as usize] - max) as f64).exp();
        }
    }
    (gathered / z) as f32
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn make_logits(seed: u64, vocab: usize) -> Vec<f32> {
    // Simple LCG so the tests are deterministic without a heavy RNG dep.
    let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    let mut out = Vec::with_capacity(vocab);
    for _ in 0..vocab {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 11) & ((1 << 24) - 1)) as f32 / ((1u32 << 24) as f32);
        // Map [0, 1) → [-4, 4) — span typical of post-temperature logits.
        out.push(u * 8.0 - 4.0);
    }
    out
}

// ---------------------------------------------------------------------------
// Entropy kernel
// ---------------------------------------------------------------------------

#[test]
fn entropy_kernel_matches_cpu_reference() {
    if !cuda_available() {
        eprintln!("cuda_available() == false; skipping (run on host with nvcc + GPU)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context on device 0");
    let stream = ctx.default_stream();

    for &vocab in &[1024usize, 16_000, 32_000, 151_936] {
        for seed in 0..3u64 {
            let host: Vec<f32> = make_logits(seed.wrapping_mul(7).wrapping_add(vocab as u64), vocab);
            let dev: CudaSlice<f32> = stream
                .memcpy_stod(&host)
                .expect("host -> device copy");
            let out_dev: CudaSlice<f32> = stream
                .memcpy_stod(&[0.0_f32])
                .expect("alloc + zero output");

            let (logits_ptr, _g0) = dev.device_ptr(&stream);
            let (out_ptr, _g1) = out_dev.device_ptr(&stream);
            unsafe {
                launch_entropy(
                    logits_ptr as *const c_void,
                    vocab,
                    Dtype::F32,
                    out_ptr as *mut f32,
                )
                .expect("launch_entropy");
            }
            stream.synchronize().expect("sync");

            let mut got = [0.0_f32; 1];
            stream.memcpy_dtoh(&out_dev, &mut got).expect("dtoh");
            let want = cpu_entropy(&host);

            let abs_err = (got[0] - want).abs();
            assert!(
                abs_err < 1e-3,
                "entropy mismatch at vocab={vocab} seed={seed}: cuda={} cpu={} diff={abs_err}",
                got[0], want,
            );
        }
    }
}

#[test]
fn eat_kernel_matches_cpu_reference() {
    if !cuda_available() {
        eprintln!("cuda_available() == false; skipping");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.default_stream();
    let end_ids_host: Vec<i32> = vec![1, 17, 256, 1024, 8000];

    for &vocab in &[16_000usize, 32_000, 151_936] {
        let host = make_logits(0xC0FFEE ^ vocab as u64, vocab);
        let dev_logits = stream.memcpy_stod(&host).expect("logits dtoh");
        let dev_end_ids = stream.memcpy_stod(&end_ids_host).expect("end_ids dtoh");
        let out_dev: CudaSlice<f32> = stream.memcpy_stod(&[0.0_f32]).expect("alloc out");

        let (lp, _g0) = dev_logits.device_ptr(&stream);
        let (ep, _g1) = dev_end_ids.device_ptr(&stream);
        let (op, _g2) = out_dev.device_ptr(&stream);
        unsafe {
            launch_eat(
                lp as *const c_void,
                vocab,
                Dtype::F32,
                ep as *const i32,
                end_ids_host.len(),
                op as *mut f32,
            )
            .expect("launch_eat");
        }
        stream.synchronize().expect("sync");

        let mut got = [0.0_f32; 1];
        stream.memcpy_dtoh(&out_dev, &mut got).expect("dtoh");
        let want = cpu_eat(&host, &end_ids_host);

        let abs_err = (got[0] - want).abs();
        assert!(
            abs_err < 1e-4,
            "EAT mismatch at vocab={vocab}: cuda={} cpu={} diff={abs_err}",
            got[0], want,
        );
        // Sanity bound.
        assert!(got[0] >= -1e-5 && got[0] <= 1.0 + 1e-5);
    }
}

#[test]
fn entropy_kernel_uniform_returns_log_v() {
    if !cuda_available() {
        eprintln!("cuda_available() == false; skipping");
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let vocab = 8192usize;
    let host = vec![0.0_f32; vocab];
    let dev = stream.memcpy_stod(&host).unwrap();
    let out_dev: CudaSlice<f32> = stream.memcpy_stod(&[0.0_f32]).unwrap();
    let (lp, _g0) = dev.device_ptr(&stream);
    let (op, _g1) = out_dev.device_ptr(&stream);
    unsafe {
        launch_entropy(lp as *const c_void, vocab, Dtype::F32, op as *mut f32).unwrap();
    }
    stream.synchronize().unwrap();
    let mut got = [0.0_f32; 1];
    stream.memcpy_dtoh(&out_dev, &mut got).unwrap();
    let want = (vocab as f32).ln();
    assert!((got[0] - want).abs() < 1e-3, "uniform: cuda={} want log(V)={}", got[0], want);
}
