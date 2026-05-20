//! # meridian-kernels
//!
//! Thin safe Rust wrappers around the Meridian entropy-probe CUDA kernels.
//!
//! By default this crate builds against a C stub that returns
//! [`KernelError::Unavailable`] for every entry point, so the workspace builds
//! cleanly without `nvcc`. Build with `--features cuda` to link the real
//! kernels — the safe surface is identical either way.
//!
//! ## Stability
//!
//! Pre-1.0. The C ABI (`meridian_entropy_launch`, `meridian_eat_launch`) is
//! the source of truth and is what downstream forks should consume directly
//! if they want to avoid Rust as a dependency.

#![doc(html_root_url = "https://docs.rs/meridian-kernels/0.1.0")]
#![warn(missing_docs, missing_debug_implementations, unreachable_pub)]

pub mod ffi;

#[cfg(feature = "nixl")]
pub mod nixl;

use core::ffi::c_void;
use core::ptr;

pub use crate::ffi::Dtype;

/// Errors that can occur when launching a kernel.
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    /// The crate was built without CUDA support, or the runtime libraries are
    /// missing.
    #[error("CUDA kernels unavailable (build without `cuda` feature or runtime missing)")]
    Unavailable,

    /// The CUDA launch returned a non-zero error code.
    #[error("CUDA kernel launch failed: code {0}")]
    Launch(i32),

    /// One of the caller-supplied pointers was null.
    #[error("null pointer passed to kernel: {0}")]
    NullPointer(&'static str),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, KernelError>;

/// Launch the Shannon-entropy kernel on a `vocab_size`-length logits array
/// resident on device memory.
///
/// # Safety
///
/// `logits` must point to at least `vocab_size` elements of the type indicated
/// by `dtype`, and `out_entropy` must point to one writable `f32` — both in
/// CUDA device memory. The caller is responsible for stream synchronisation.
pub unsafe fn launch_entropy(
    logits: *const c_void,
    vocab_size: usize,
    dtype: Dtype,
    out_entropy: *mut f32,
) -> Result<()> {
    if logits.is_null() {
        return Err(KernelError::NullPointer("logits"));
    }
    if out_entropy.is_null() {
        return Err(KernelError::NullPointer("out_entropy"));
    }
    let rc = unsafe { ffi::meridian_entropy_launch(logits, vocab_size, dtype as i32, out_entropy) };
    match rc {
        0 => Ok(()),
        -1 => Err(KernelError::Unavailable),
        code => Err(KernelError::Launch(code)),
    }
}

/// Launch the EAT kernel: sum of softmax over `think_end_ids`.
///
/// # Safety
///
/// All four pointer arguments must satisfy the same residency and lifetime
/// constraints as in [`launch_entropy`].
pub unsafe fn launch_eat(
    logits: *const c_void,
    vocab_size: usize,
    dtype: Dtype,
    think_end_ids: *const i32,
    n_end_ids: usize,
    out_eat: *mut f32,
) -> Result<()> {
    if logits.is_null() {
        return Err(KernelError::NullPointer("logits"));
    }
    if think_end_ids.is_null() && n_end_ids != 0 {
        return Err(KernelError::NullPointer("think_end_ids"));
    }
    if out_eat.is_null() {
        return Err(KernelError::NullPointer("out_eat"));
    }
    let rc = unsafe {
        ffi::meridian_eat_launch(
            logits,
            vocab_size,
            dtype as i32,
            think_end_ids,
            n_end_ids,
            out_eat,
        )
    };
    match rc {
        0 => Ok(()),
        -1 => Err(KernelError::Unavailable),
        code => Err(KernelError::Launch(code)),
    }
}

/// Returns whether the linked backend is the real CUDA build (`true`) or the
/// stub (`false`). Useful for telemetry and for skipping GPU-required tests.
#[must_use]
pub fn cuda_available() -> bool {
    // Probe via a zero-sized call: stub returns -1 unconditionally.
    let mut out = 0f32;
    let rc = unsafe {
        ffi::meridian_entropy_launch(ptr::null::<c_void>().wrapping_add(1), 0, 0, &mut out)
    };
    rc != -1
}
