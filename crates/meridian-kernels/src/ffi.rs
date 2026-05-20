//! Raw FFI declarations.
//!
//! This is the entire `unsafe` surface of `meridian-kernels`. Everything else
//! in the crate wraps these in safe Rust.
//!
//! ## Two compilation modes
//!
//! - **`--features cuda`**: `extern "C"` declarations bind to symbols
//!   provided by the CMake-built `meridian_kernels_cuda` static library.
//! - **Default**: stub implementations live in this file (`#[no_mangle]`
//!   functions) and return `-1` (kErrUnavailable) without doing any work.
//!   No external linkage is required.
//!
//! The safe wrappers in `src/lib.rs` do not need to branch on feature flag
//! — both modes expose the same symbol names and the same calling
//! convention.

use core::ffi::{c_int, c_void};

/// `MeridianDtype` enum mirror.
#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// `float32`
    F32 = 0,
    /// `bfloat16`
    Bf16 = 1,
    /// `float16`
    F16 = 2,
}

// ---------------------------------------------------------------------------
// CUDA build: bind to symbols from the CMake-built static library.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// Launch the Shannon-entropy CUDA kernel. See `entropy_probe.cuh`.
    pub fn meridian_entropy_launch(
        logits: *const c_void,
        vocab_size: usize,
        dtype: c_int,
        out_entropy: *mut f32,
    ) -> c_int;

    /// Launch the EAT CUDA kernel. See `entropy_probe.cuh`.
    pub fn meridian_eat_launch(
        logits: *const c_void,
        vocab_size: usize,
        dtype: c_int,
        think_end_ids: *const i32,
        n_end_ids: usize,
        out_eat: *mut f32,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Default (no CUDA): provide stub symbols from Rust. Returns -1 so the safe
// wrapper translates to `KernelError::Unavailable`.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
mod stubs {
    use super::{c_int, c_void};

    /// # Safety
    ///
    /// Stub: ignores all pointer arguments and returns `-1`. Safe to call
    /// with any (including null) values; downstream code is expected to
    /// interpret `-1` as `KernelError::Unavailable`.
    #[allow(clippy::missing_const_for_fn)]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn meridian_entropy_launch(
        _logits: *const c_void,
        _vocab_size: usize,
        _dtype: c_int,
        _out_entropy: *mut f32,
    ) -> c_int {
        -1
    }

    /// # Safety
    ///
    /// Stub: ignores all pointer arguments and returns `-1`. Safe to call
    /// with any (including null) values; downstream code is expected to
    /// interpret `-1` as `KernelError::Unavailable`.
    #[allow(clippy::missing_const_for_fn)]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn meridian_eat_launch(
        _logits: *const c_void,
        _vocab_size: usize,
        _dtype: c_int,
        _think_end_ids: *const i32,
        _n_end_ids: usize,
        _out_eat: *mut f32,
    ) -> c_int {
        -1
    }
}

#[cfg(not(feature = "cuda"))]
pub use stubs::{meridian_eat_launch, meridian_entropy_launch};
