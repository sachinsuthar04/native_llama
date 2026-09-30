// Unit tests for the KV prefix-reuse planner (shared_cpp/common/nl-kv-reuse.h).
//
// This is the piece that decides whether a generation re-prefills thousands of
// tokens or a handful, and a wrong answer here means generating from a corrupted
// context — so it is worth testing away from a phone.
//
// Build and run (no llama.cpp link needed; the header only uses llama_token):
//
//   c++ -std=c++17 -I ../ios/shared_cpp/include -I ../ios/shared_cpp/ggml/include \
//       -I ../ios/shared_cpp/common nl_kv_reuse_test.cpp -o /tmp/nl_kv_reuse_test \
//   && /tmp/nl_kv_reuse_test

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nl-kv-reuse.h"

static int g_failures = 0;

static void check(bool ok, const char * name, const char * detail) {
    if (ok) {
        std::printf("  ok    %s\n", name);
    } else {
        std::printf("  FAIL  %s — %s\n", name, detail);
        g_failures++;
    }
}

static void expect(const char * name,
                   const nl_prefix_plan & got,
                   size_t n_reuse, bool needs_trim, bool needs_clear) {
    const bool ok = got.n_reuse == n_reuse
                 && got.needs_trim == needs_trim
                 && got.needs_clear == needs_clear;
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "expected {reuse=%zu trim=%d clear=%d}, got {reuse=%zu trim=%d clear=%d}",
                  n_reuse, (int) needs_trim, (int) needs_clear,
                  got.n_reuse, (int) got.needs_trim, (int) got.needs_clear);
    check(ok, name, detail);
}

using toks = std::vector<llama_token>;

int main() {
    std::printf("nl_plan_prefix_reuse\n");

    // --- The case that matters: an append-only conversation ---------------
    // Cache holds the previous prompt plus the reply the model generated. The
    // next prompt is all of that plus a turn terminator and the new user turn.
    // Nothing to evict; decode resumes at the end of the cache.
    expect("pure append reuses the whole cache, no memory op",
           nl_plan_prefix_reuse(toks{1, 2, 3, 4}, toks{1, 2, 3, 4, 5, 6, 7}, true),
           /*n_reuse=*/4, /*trim=*/false, /*clear=*/false);

    expect("append of a single token",
           nl_plan_prefix_reuse(toks{1, 2, 3}, toks{1, 2, 3, 4}, true),
           3, false, false);

    // --- Divergence: needs an eviction the caller may not be granted -------
    expect("divergent tail asks for a trim",
           nl_plan_prefix_reuse(toks{1, 2, 3, 9, 9}, toks{1, 2, 3, 4, 5}, true),
           3, true, false);

    expect("prompt is a strict prefix of the cache still needs a trim",
           nl_plan_prefix_reuse(toks{1, 2, 3, 4, 5}, toks{1, 2, 3}, true),
           2, true, false);

    // --- Nothing shared → full prefill ------------------------------------
    expect("no common prefix clears",
           nl_plan_prefix_reuse(toks{7, 8, 9}, toks{1, 2, 3}, true),
           0, false, true);

    expect("first call (empty cache) clears",
           nl_plan_prefix_reuse(toks{}, toks{1, 2, 3}, true),
           0, false, true);

    // --- The veto: vision and speculative decoding -------------------------
    expect("may_reuse=false always clears, even on a perfect append",
           nl_plan_prefix_reuse(toks{1, 2, 3}, toks{1, 2, 3, 4}, false),
           0, false, true);

    // --- Logits edge case --------------------------------------------------
    // If the whole prompt were reused there would be no fresh logits to sample
    // the first output token from, so one token must always be re-decoded.
    expect("identical prompt still decodes one token",
           nl_plan_prefix_reuse(toks{1, 2, 3}, toks{1, 2, 3}, true),
           2, true, false);

    expect("single-token prompt fully cached falls back to a clear",
           nl_plan_prefix_reuse(toks{1}, toks{1}, true),
           0, false, true);

    // --- Degenerate inputs -------------------------------------------------
    expect("empty prompt clears",
           nl_plan_prefix_reuse(toks{1, 2}, toks{}, true),
           0, false, true);

    expect("both empty clears",
           nl_plan_prefix_reuse(toks{}, toks{}, true),
           0, false, true);

    // --- Invariants --------------------------------------------------------
    check(!(nl_plan_prefix_reuse(toks{1, 2, 3}, toks{1, 2, 3, 4}, true).needs_trim &&
            nl_plan_prefix_reuse(toks{1, 2, 3}, toks{1, 2, 3, 4}, true).needs_clear),
          "trim and clear are mutually exclusive", "both set");

    // A plan must never ask to resume beyond what the cache actually holds.
    {
        const toks cached{1, 2, 3, 4};
        const toks prompt{1, 2, 3, 4, 5, 6};
        const auto p = nl_plan_prefix_reuse(cached, prompt, true);
        check(p.n_reuse <= cached.size(), "n_reuse never exceeds the cache", "resume past end of cache");
        check(p.n_reuse < prompt.size(), "n_reuse always leaves a token to decode", "no logits would be produced");
    }

    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "PASS" : "FAIL",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
