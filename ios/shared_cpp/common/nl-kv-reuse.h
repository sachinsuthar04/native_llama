// KV-cache prefix reuse planning — native_llama's own code, not upstream
// llama.cpp. It lives under shared_cpp/common because that directory is on the
// include path of BOTH build systems (the iOS podspec's source_files and
// Android's target_include_directories), so the iOS bridge and the Android JNI
// layer can share one implementation instead of drifting apart. The `nl-` prefix
// marks it as ours; keep it when re-vendoring llama.cpp.
//
// THE PROBLEM
//
// A chat turn re-sends the whole conversation. Without reuse, every turn
// re-prefills thousands of tokens the KV cache already holds, and that prefill —
// not decoding — dominates time-to-first-token.
//
// THE CONSTRAINT
//
// You cannot always rewind a KV cache. For recurrent and hybrid architectures
// (Mamba, RWKV, and LFM2's shortconv blocks — see hparams.recurrent_layer_arr)
// a layer's state is a rolling window over the tokens it has seen, not
// per-position storage. There is nothing to truncate back to, so
// llama_memory_recurrent::seq_rm refuses a partial removal that includes the
// final position and returns false rather than corrupt itself.
//
// That leaves exactly one universally safe form of reuse: PURE APPEND, where the
// cached tokens are a strict prefix of the new prompt and nothing needs
// evicting. Conveniently that is the shape an append-only conversation produces,
// which is why prompts should be laid out most-stable-content-first.
//
// Divergent reuse (cache holds tokens the new prompt doesn't) is still attempted
// — it is a large win on plain transformer models — but the caller must treat a
// false from seq_rm as "fall back to a full clear", never as something to ignore.

#pragma once

#include <cstddef>
#include <vector>

#include "llama.h"

/// What the caller must do to the memory before decoding, and where decoding
/// should start.
struct nl_prefix_plan {
    /// Number of leading prompt tokens already in the cache. Decoding starts at
    /// this position; 0 means prefill the whole prompt.
    size_t n_reuse = 0;

    /// Caller must evict the cache tail past `n_reuse` — llama_memory_seq_rm(
    /// mem, 0, n_reuse, -1) — before decoding. If that returns false the
    /// architecture cannot be rewound: clear the memory and prefill from 0.
    bool needs_trim = false;

    /// Caller must llama_memory_clear() before decoding. Mutually exclusive with
    /// `needs_trim`.
    bool needs_clear = false;
};

/// Length of the longest common prefix of `a` and `b`.
inline size_t nl_common_prefix_len(const std::vector<llama_token> & a,
                                   const std::vector<llama_token> & b) {
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

/// Decides how much of `cached` can serve as a prefix for `prompt`.
///
/// `may_reuse` is the caller's veto for paths whose cache movements this vector
/// cannot mirror — vision/MTMD (which owns its own position cursor) and
/// speculative decoding (whose accept/reject rollbacks move the cache
/// underneath us).
inline nl_prefix_plan nl_plan_prefix_reuse(const std::vector<llama_token> & cached,
                                           const std::vector<llama_token> & prompt,
                                           bool may_reuse) {
    nl_prefix_plan plan;

    if (!may_reuse || cached.empty() || prompt.empty()) {
        // Start from a known-clean context rather than trusting whatever a
        // previous path left behind.
        plan.needs_clear = true;
        return plan;
    }

    size_t n = nl_common_prefix_len(cached, prompt);

    // At least one token must be decoded this call, or there are no fresh logits
    // to sample the first output token from. This fires when the prompt is
    // wholly contained in the cache — e.g. a regenerate of an identical prompt.
    if (n == prompt.size()) n--;

    if (n == 0) {
        plan.needs_clear = true;
        return plan;
    }

    plan.n_reuse = n;

    // n == cached.size() is the pure-append case: nothing to evict, legal on
    // every architecture. Otherwise the cache holds a diverging tail that has to
    // go, which the caller must be ready to see refused.
    plan.needs_trim = (n < cached.size());
    return plan;
}
