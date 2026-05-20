// Public C-ABI surface of the Meridian entropy probe CUDA kernels.
//
// Sprint 0: signatures only — the corresponding .cu files compile to empty
// kernels so the static library links cleanly. The full kernel bodies from
// the playbook (§3.5) land in Sprint 1, behind the same ABI.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// dtype values for the type-erased logits pointer.
enum MeridianDtype {
    kMeridianDtypeF32  = 0,
    kMeridianDtypeBf16 = 1,
    kMeridianDtypeF16  = 2,
};

// Launch the Shannon-entropy kernel on logits[vocab_size].
// Writes the scalar entropy (nats) into *out_entropy on device memory.
//
// Returns 0 on success, negative on launch failure.
int meridian_entropy_launch(
    const void* logits,
    size_t      vocab_size,
    int         dtype,
    float*      out_entropy
);

// Launch the EAT kernel: sum of softmax probability over the given
// think_end_ids set. Out value written into *out_eat (device memory).
//
// Returns 0 on success, negative on launch failure.
int meridian_eat_launch(
    const void*    logits,
    size_t         vocab_size,
    int            dtype,
    const int32_t* think_end_ids,
    size_t         n_end_ids,
    float*         out_eat
);

#ifdef __cplusplus
} // extern "C"
#endif
