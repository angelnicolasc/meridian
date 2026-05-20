//! D28 — verify the default (no-`cuda`) build returns `KernelError::Unavailable`
//! from every kernel entry point, and that `cuda_available()` reflects that.
//!
//! Trivial but load-bearing: the entire crate is designed so that downstream
//! consumers can depend on it without a CUDA toolchain. A regression here
//! would silently break that contract.

#![cfg(not(feature = "cuda"))]

use core::ffi::c_void;

use meridian_kernels::{Dtype, KernelError, cuda_available, launch_eat, launch_entropy};

#[test]
fn cuda_available_reports_stub_backend() {
    assert!(
        !cuda_available(),
        "default build must report no CUDA backend",
    );
}

#[test]
fn launch_entropy_returns_unavailable() {
    let dummy: f32 = 0.0;
    let mut out: f32 = 0.0;
    let logits = (&raw const dummy).cast::<c_void>();
    let res = unsafe { launch_entropy(logits, 1, Dtype::F32, &raw mut out) };
    match res {
        Err(KernelError::Unavailable) => {}
        Err(other) => panic!("expected Unavailable, got {other:?}"),
        Ok(()) => panic!("stub must not report success"),
    }
}

#[test]
fn launch_eat_returns_unavailable() {
    let dummy_logits: f32 = 0.0;
    let dummy_ids: i32 = 0;
    let mut out: f32 = 0.0;
    let res = unsafe {
        launch_eat(
            (&raw const dummy_logits).cast::<c_void>(),
            1,
            Dtype::F32,
            &raw const dummy_ids,
            1,
            &raw mut out,
        )
    };
    assert!(
        matches!(res, Err(KernelError::Unavailable)),
        "expected Unavailable, got {res:?}",
    );
}

#[test]
fn null_pointer_guards_fire_before_ffi() {
    let mut out: f32 = 0.0;
    let res = unsafe {
        launch_entropy(core::ptr::null::<c_void>(), 1, Dtype::F32, &raw mut out)
    };
    assert!(matches!(res, Err(KernelError::NullPointer(_))));
}
