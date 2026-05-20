//! Build script for `meridian-kernels`.
//!
//! Two build modes:
//!
//! 1. **Default (no CUDA)** — does nothing. The Rust source provides safe
//!    stub implementations of every FFI entry point gated behind
//!    `cfg(not(feature = "cuda"))`, so the workspace builds cleanly on any
//!    host without `nvcc` or even a C toolchain.
//! 2. **`--features cuda`** — invokes CMake against `crates/meridian-kernels/`
//!    to compile the real CUDA kernels and link them as a static library.
//!
//! The default mode is the common path. The `cuda` feature is opt-in for
//! GPU CI and production builds.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=cuda/");

    let cuda_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();

    if cuda_enabled {
        build_cuda();
    }
    // Default: no native build. The Rust source provides stub symbols.
}

fn build_cuda() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let dst = cmake::Config::new(&manifest_dir)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("MERIDIAN_BUILD_CUDA", "ON")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=meridian_kernels_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
