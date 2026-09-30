#import "LlamaBridge.h"
#import <llama.h>
#import "ggml-backend.h"
#import <vector>
#import <string>
#import <exception>
#import <sys/stat.h>

// --- MTMD Headers ---
#import "mtmd.h"
#import "mtmd-helper.h"
#import "chat.h"  // common_chat_* — Jinja chat-template formatting (like mtmd-cli)
#import "nl-kv-reuse.h"  // KV prefix reuse planning, shared with the Android JNI layer

/// Samples one token, converting a grammar failure into end-of-generation.
///
/// A GBNF grammar throws `std::runtime_error` if it is ever asked to accept a
/// token it has ruled out, and `llama_sampler_sample` accepts internally — so
/// the throw comes out of the sample call. Uncaught it reaches
/// `ggml_uncaught_exception` and kills the app mid-reflection. Ending the stream
/// instead leaves the caller with whatever was produced, which its own parser
/// and fallbacks already handle. Sets `*failed` and returns 0 on failure.
static llama_token nl_sample_or_stop(llama_sampler * smpl, llama_context * lctx, int32_t idx, bool * failed) {
    try {
        return llama_sampler_sample(smpl, lctx, idx);
    } catch (const std::exception & e) {
        NSLog(@"[LlamaBridge] Generation stopped by sampler: %s", e.what());
        *failed = true;
        return 0;
    }
}

@implementation LlamaBridge {
    llama_model *model;
    llama_context *ctx;
    llama_model *model_draft;
    llama_context *ctx_draft;
    mtmd_context *mtmd_ctx;
    bool stop_generation;
    volatile bool is_generating;
    int32_t active_n_ctx; // resolved context window of the live ctx (for resetContext)

    /// The exact token sequence currently held in the KV cache at seq 0,
    /// positions 0..cached_tokens.size()-1 — prompt tokens plus every token
    /// decoded back into the context during generation.
    ///
    /// This is what makes prefix reuse possible: on the next call we can compare
    /// the new prompt against it and skip re-prefilling the shared head. Kept
    /// exact, because a wrong entry here means silently generating from a
    /// corrupted context, which is far worse than a slow prefill. Any code path
    /// that mutates the cache in a way we can't mirror must call
    /// [self invalidatePrefixCache].
    std::vector<llama_token> cached_tokens;
}

- (void)invalidatePrefixCache {
    cached_tokens.clear();
}

+ (instancetype)shared {
    static LlamaBridge *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        // Disable ggml-metal's new "tensor API" matmul path. It is enabled by
        // default on A19/M5-class GPUs but miscomputes the Qwen2.5-VL vision
        // encoder there, making on-device document extraction return all-empty
        // fields. Forcing the mature simdgroup path (what every other GPU uses)
        // restores correct output at full GPU speed. Must be set before the Metal
        // device is initialized (i.e. before llama_backend_init). Harmless on
        // non-Metal backends (Android/Vulkan just ignores it).
        setenv("GGML_METAL_TENSOR_DISABLE", "1", 1);
        llama_backend_init();
        ggml_backend_load_all();
        model = nullptr;
        ctx = nullptr;
        model_draft = nullptr;
        ctx_draft = nullptr;
        mtmd_ctx = nullptr;
        stop_generation = false;
        is_generating = false;
        active_n_ctx = 0;
    }
    return self;
}

- (BOOL)initModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads nGpuLayers:(int)nGpuLayers {
    const char *path = [modelPath UTF8String];

    uint64_t physicalMemory = [[NSProcessInfo processInfo] physicalMemory];
    double memoryGB = physicalMemory / (1024.0 * 1024.0 * 1024.0);

    struct stat stat_buf;
    double fileSizeGB = 0;
    if (stat(path, &stat_buf) == 0) {
        fileSizeGB = stat_buf.st_size / (1024.0 * 1024.0 * 1024.0);
    }

    if (fileSizeGB > 0 && fileSizeGB > (memoryGB * 0.65)) {
        NSLog(@"NATIVE_LLAMA: RAM SHIELD - Model size (%.2f GB) exceeds safe limits for device RAM (%.2f GB). Aborting load.", fileSizeGB, memoryGB);
        return NO;
    }

    [self unload];

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = nGpuLayers;
    mparams.use_mmap = false;

    model = llama_model_load_from_file(path, mparams);
    if (model == nullptr) return NO;

    auto cparams = llama_context_default_params();
    cparams.n_threads = nThreads > 0 ? nThreads : 4;
    cparams.embeddings = true;
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    cparams.n_batch = 128;

    if (nCtx > 0) {
        cparams.n_ctx = nCtx;
    } else {
        int32_t dynamic_n_ctx = 4096;
        if (memoryGB >= 7.5) dynamic_n_ctx = 8192;
        if (memoryGB >= 11.5) dynamic_n_ctx = 16384;
        cparams.n_ctx = dynamic_n_ctx;
    }

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        model = nullptr;
        return NO;
    }
    active_n_ctx = cparams.n_ctx; // remember for a later resetContext
    return YES;
}

// Recreate ONLY the llama context, keeping the model weights and the mtmd vision
// projector resident. A fresh context resets the KV cache and the M-RoPE
// position state, so the next generation (e.g. the next page image) starts clean
// WITHOUT the multi-GB weight + projector reload that dispose + initModel costs.
// This is the safe, cheap way to read several images in sequence: llama.cpp's
// per-generation llama_memory_clear does not reliably reset M-RoPE positions on
// a reused vision context after a large/aborted decode, which corrupts the
// backend; a brand-new context does.
- (BOOL)resetContext:(int)nCtx {
    if (model == nullptr) return NO; // nothing loaded — caller falls back
    if (is_generating) return NO;    // never free a context mid-decode

    if (ctx != nullptr) {
        llama_free(ctx);
        ctx = nullptr;
    }

    // Rebuild the context params EXACTLY as initModel did (same KV quantisation,
    // batch, embeddings), reusing the resolved window unless the caller overrides.
    auto cparams = llama_context_default_params();
    cparams.n_threads = 4;
    cparams.embeddings = true;
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    cparams.n_batch = 128;
    cparams.n_ctx = nCtx > 0 ? nCtx : (active_n_ctx > 0 ? active_n_ctx : 4096);

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) return NO;
    active_n_ctx = cparams.n_ctx;
    [self invalidatePrefixCache]; // the KV that cache described no longer exists
    return YES;
}

- (BOOL)initVision:(NSString *)mmprojPath {
    const char *path = [mmprojPath UTF8String];
    NSLog(@"Initializing Vision model from: %s", path);

    if (mtmd_ctx) { mtmd_free(mtmd_ctx); mtmd_ctx = nullptr; }

    if (model == nullptr) {
        NSLog(@"Error: Main text model must be initialized before the vision projector.");
        return NO;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    // Vision encoder on the GPU (Metal). The real fix for the all-empty on-device
    // extraction is NOT here — it's the GGML_METAL_TENSOR_DISABLE setenv in -init.
    // On A19/M5-class GPUs ggml-metal enables a brand-new "tensor API" matmul path
    // that miscomputes the Qwen2.5-VL vision graph: the image decodes, the LLM even
    // sees the coarse page structure, but the embeddings are garbage so every field
    // reads back "". The mature simdgroup path (every other GPU, incl. Macs and the
    // A19 once the tensor API is off) is correct. Verified: identical vendored code
    // reads this exact Form 16 correctly on macOS Metal; the device (Apple A19 Pro,
    // has_tensor=true) is the only config that fails.
    // Do NOT switch this to CPU (use_gpu=false) as a workaround: it reads correctly
    // but takes >15 min per page on the device.
    mtmd_params.use_gpu = true;
    // Disable flash attention in the vision encoder. clip's attention defaults to
    // AUTO→ENABLED, i.e. ggml_flash_attn_ext, whose Metal kernel is the most
    // complex op in the vision graph and the prime suspect for the A19's garbage
    // embeddings (the mundane simdgroup matmul path is already correct once the
    // tensor API is off). Forcing plain softmax+matmul attention here trades a
    // little speed for the robust, well-exercised kernels. Metal-only concern;
    // Vulkan/CPU are unaffected by the value.
    mtmd_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // Cap vision tokens per image. Full-page document reads (Qwen2.5-VL) need
    // enough vision tokens to keep fine print legible; 1536 handles a dense A4
    // page without tiling. (image_min_tokens keeps small crops legible.)
    mtmd_params.image_max_tokens = 1536;
    mtmd_params.image_min_tokens = 256;

    mtmd_ctx = mtmd_init_from_file(path, model, mtmd_params);

    if (mtmd_ctx == nullptr) {
        NSLog(@"Failed to load MTMD mmproj model: %s", path);
        return NO;
    }

    NSLog(@"Vision model initialized successfully via MTMD");
    return YES;
}

- (BOOL)initDraftModel:(NSString *)modelPath nCtx:(int)nCtx nThreads:(int)nThreads nGpuLayers:(int)nGpuLayers {
    const char *path = [modelPath UTF8String];
    if (ctx_draft) { llama_free(ctx_draft); ctx_draft = nullptr; }
    if (model_draft) { llama_model_free(model_draft); model_draft = nullptr; }

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = nGpuLayers;
    mparams.use_mmap = false;

    model_draft = llama_model_load_from_file(path, mparams);
    if (model_draft == nullptr) return NO;

    auto cparams = llama_context_default_params();
    cparams.n_threads = nThreads > 0 ? nThreads : 4;
    cparams.embeddings = true;
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
    cparams.n_batch = 32;

    if (nCtx > 0) {
        cparams.n_ctx = nCtx;
    } else {
        uint64_t physicalMemory = [[NSProcessInfo processInfo] physicalMemory];
        double memoryGB = physicalMemory / (1024.0 * 1024.0 * 1024.0);
        int32_t dynamic_n_ctx = 4096;
        if (memoryGB >= 7.5) dynamic_n_ctx = 8192;
        if (memoryGB >= 11.5) dynamic_n_ctx = 16384;
        cparams.n_ctx = dynamic_n_ctx;
    }

    ctx_draft = llama_init_from_model(model_draft, cparams);
    return ctx_draft != nullptr;
}

- (NSArray<NSNumber *> *)getEmbedding:(NSString *)text {
    if (ctx == nullptr || model == nullptr) return nil;
    // Generation turns this off (see startGenerationWithRoles:) because it makes
    // every prompt token an output. Turn it back on here, where the embeddings
    // are the entire point. The KV this leaves behind is cleared below anyway.
    llama_set_embeddings(ctx, true);

    // Clear the KV before embedding, and invalidate the prefix cache that
    // described it. Android's getEmbedding has done this since it was written
    // ("CRITICAL FIX: Clear the KV cache so embeddings don't stack up
    // indefinitely!"); iOS never did, so every embedding call left its tokens
    // in the cache and the next one embedded on top of them. Wrong vectors, and
    // a context that fills up for no reason.
    llama_memory_clear(llama_get_memory(ctx), true);
    [self invalidatePrefixCache];

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = [text UTF8String];

    int n_tokens = -llama_tokenize(vocab, prompt, (int)strlen(prompt), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt, (int)strlen(prompt), tokens.data(), (int)tokens.size(), true, true);

    uint32_t n_batch = llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int n_eval = 0;
    while (n_eval < (int)tokens.size()) {
        int n_chunk = MIN((int)tokens.size() - n_eval, n_batch);
        batch.n_tokens = 0;
        for (int i = 0; i < n_chunk; ++i) {
            batch.token[i] = tokens[n_eval + i];
            batch.pos[i] = n_eval + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
            batch.n_tokens++;
        }
        if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); return nil; }
        n_eval += n_chunk;
    }

    float * embd = llama_get_embeddings(ctx);
    if (embd == nullptr) { llama_batch_free(batch); return nil; }

    int n_embd = llama_model_n_embd(model);
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:n_embd];
    for (int i = 0; i < n_embd; i++) { [result addObject:@(embd[i])]; }

    llama_batch_free(batch);
    return result;
}

struct GenerationGuard {
    volatile bool *flag;
    GenerationGuard(volatile bool *f) : flag(f) { *flag = true; }
    ~GenerationGuard() { if (flag) *flag = false; }
};

- (void)startGenerationWithRoles:(NSArray<NSString *> *)roles contents:(NSArray<NSString *> *)contents mediaPaths:(NSArray<NSString *> *)mediaPaths temperature:(float)temperature topK:(int)topK topP:(float)topP repeatPenalty:(float)repeatPenalty penaltyLastN:(int)penaltyLastN freqPenalty:(float)freqPenalty presencePenalty:(float)presencePenalty grammar:(NSString *)grammar onToken:(void (^)(NSString *))onToken {
    if (ctx == nullptr || model == nullptr) return;

    // One context cannot decode twice at once. A second llama_decode while the
    // first is in flight fails the Metal command buffer, leaves the backend in
    // an unrecoverable error state, and aborts the PROCESS on
    // GGML_ASSERT(out_ids.size() == n_outputs) — a hard crash with a backtrace
    // that points at the assert rather than at the overlap that caused it.
    //
    // Observed for real: a caller kicked off a cache-warming generation without
    // awaiting it, the user tapped, and the app died. Refusing the second call
    // is not a fix for that caller's bug — they still must serialise — but a
    // dropped generation beats a dead app, and the log line names the cause.
    if (is_generating) {
        fprintf(stderr, "nl_generate: called while already generating - refused. "
                        "Serialise your calls; one context cannot decode twice at once.\n");
        // Must be the end-of-stream sentinel, not nil: the Swift layer drops a
        // nil token (`guard let token = token else { return }`) and only closes
        // the Dart stream on this exact string. Returning nil here would swap a
        // crash for a caller hanging forever, which is worse.
        if (onToken) onToken(@"__END_OF_STREAM__");
        return;
    }

    GenerationGuard guard(&is_generating);
    stop_generation = false;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // --- CRITICAL FIX: Only use the draft model if NO media is attached ---
    bool use_draft = (ctx_draft != nullptr && (mediaPaths == nil || mediaPaths.count == 0));

    // Speculative decoding and a GBNF grammar cannot share one sampler chain.
    // Drafting samples from the draft context through this same chain, which
    // advances the grammar's stack for tokens that verification may then throw
    // away — and there is no API to rewind grammar state. The result is a
    // grammar desynchronised from the real token stream, which aborts the
    // process the first time it is asked to accept a token it has ruled out.
    // A grammared reply is worth more than the speed-up.
    const bool has_grammar = (grammar != nil && grammar.length > 0);
    if (has_grammar) use_draft = false;

    // Whether this call may reuse the KV already in the cache (see
    // cached_tokens). Two paths can never reuse it:
    //
    //  • MTMD/vision — image chunks are evaluated through mtmd_helper_eval_chunks,
    //    which owns the position cursor and (for M-RoPE models) lays out
    //    positions in a way plain token indices don't describe. We can't mirror
    //    that in cached_tokens, so vision always starts clean.
    //  • Speculative decoding — the draft context would have to be kept in
    //    lockstep with the target's cache through accept/reject rollbacks. Not
    //    worth the complexity: a grammar already disables drafting, and every
    //    structured caller uses one.
    //
    // Everything else defers the decision to the text-only prefill below, which
    // reuses whatever prefix it can and clears only when it must.
    // Generation does not want embeddings, and leaving them on is expensive in
    // a way that is invisible unless you read the log.
    //
    // The context is created with cparams.embeddings = true so getEmbedding:
    // works. But with that flag set, llama_batch_allocr sees a prefill batch
    // whose interior tokens are not marked as outputs and OVERRIDES THEM ALL to
    // true (llama-batch.cpp, "embeddings required but some input tokens were not
    // marked as outputs -> overriding"). Every prompt token then gets a full
    // output computed instead of just the last one, and the output buffer grows
    // from 0.59 MiB to 75.31 MiB. On a ~1,200-token prompt that tax lands on
    // every single turn, and it is paid during prefill — the part the player
    // waits through before the first word appears.
    //
    // Toggle it off for generation and back on where embeddings are actually
    // read. Callers of getEmbedding: are unaffected.
    llama_set_embeddings(ctx, false);

    const bool has_media = (mediaPaths != nullptr && mediaPaths.count > 0);
    const bool may_reuse_prefix = !use_draft && !(mtmd_ctx != nullptr && has_media);

    if (!may_reuse_prefix) {
        // Reset context between generations. A partial seq_rm(-1,-1,-1) does NOT
        // reset the cell tails of recurrent/hybrid models (e.g. Qwen3-Next / Mamba),
        // which then trips `GGML_ASSERT(cell.has_seq_id(seq_id))` in
        // llama_memory_recurrent::find_slot on the next decode. llama_memory_clear
        // fully resets both transformer KV and recurrent state.
        llama_memory_clear(llama_get_memory(ctx), true);
        if (use_draft) llama_memory_clear(llama_get_memory(ctx_draft), true);
        [self invalidatePrefixCache];
    }

    uint32_t n_batch_size = llama_n_batch(ctx);
    if (use_draft) { n_batch_size = MIN(n_batch_size, llama_n_batch(ctx_draft)); }

    // Format the prompt with the model's OWN chat template via the Jinja
    // (common_chat) path — the same one mtmd-cli uses. The C-API
    // llama_chat_apply_template does NOT honor custom templates like
    // NuExtract3's (which frames the extraction task and enables the model's
    // thinking), which produced hallucinated, task-less output. use_jinja=true
    // fixes that. Falls back to a plain role/content concatenation if the
    // template can't be applied.
    std::string prompt_str;
    try {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        inputs.add_generation_prompt = true;
        // Thinking ON: NuExtract3 appears to need its reasoning step to actually
        // READ the page (accurate only with thinking on in testing). The Dart
        // layer bounds runaway generation with a hard token cap + stop-on-JSON,
        // so this can't ramble indefinitely.
        inputs.enable_thinking = true;
        // Assistant prefill: a trailing message with role "assistant" is NOT a
        // completed turn — it's a seed the model must CONTINUE (e.g. "{" to force
        // an immediate JSON reply from a model that would otherwise think out
        // loud). Rendering it through the template would close the turn with an
        // EOS/<|im_end|>, so instead we keep add_generation_prompt=true (prompt
        // ends at the open assistant turn) and append the seed text afterwards.
        NSUInteger n_msg = roles.count;
        std::string assistant_prefix;
        if (n_msg > 0 && [roles[n_msg - 1] isEqualToString:@"assistant"]) {
            assistant_prefix = std::string([contents[n_msg - 1] UTF8String]);
            n_msg -= 1; // don't feed the seed to the template
        }
        for (NSUInteger i = 0; i < n_msg; i++) {
            common_chat_msg m;
            m.role = [roles[i] UTF8String];
            m.content = [contents[i] UTF8String];
            inputs.messages.push_back(m);
        }
        common_chat_params cparams = common_chat_templates_apply(tmpls.get(), inputs);
        prompt_str = cparams.prompt + assistant_prefix;
    } catch (const std::exception & e) {
        NSLog(@"common_chat template apply failed (%s) — falling back", e.what());
    }
    if (prompt_str.empty()) {
        for (NSUInteger i = 0; i < roles.count; i++) {
            if (i + 1 == roles.count && [roles[i] isEqualToString:@"assistant"]) {
                // seed the assistant turn (prefill), leaving it open to continue
                prompt_str += "assistant: " + std::string([contents[i] UTF8String]);
            } else {
                prompt_str += std::string([roles[i] UTF8String]) + ": " +
                              std::string([contents[i] UTF8String]) + "\n";
            }
        }
        if (roles.count == 0 || ![roles[roles.count - 1] isEqualToString:@"assistant"]) {
            prompt_str += "assistant: ";
        }
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // Optional GBNF grammar. Added FIRST so it masks every token the grammar
    // forbids before temperature/top-k/top-p see the distribution; later in the
    // chain those samplers could pick a candidate the grammar has ruled out.
    // A grammar that fails to parse returns NULL and is treated as "no grammar"
    // — an unconstrained answer the caller can still validate beats refusing to
    // generate at all.
    if (has_grammar) {
        const llama_vocab * gvocab = llama_model_get_vocab(model);
        llama_sampler * gsmpl = llama_sampler_init_grammar(gvocab, [grammar UTF8String], "root");
        if (gsmpl != nullptr) {
            llama_sampler_chain_add(smpl, gsmpl);
        } else {
            NSLog(@"[LlamaBridge] Grammar failed to parse; continuing unconstrained");
        }
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
    // Repetition penalty is caller-configurable: structured/list output (e.g. a
    // JSON array of similar objects) is hurt by it — the repeated structural
    // tokens get penalised, so the model emits fewer items and stops early. When
    // the penalty is effectively off, skip the sampler entirely.
    if (repeatPenalty > 1.0f || freqPenalty != 0.0f || presencePenalty != 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(penaltyLastN, repeatPenalty, freqPenalty, presencePenalty));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    int n_cur = 0;
    int n_prompt_tokens_total = 0;

    // --- MTMD Evaluation ---
    if (mtmd_ctx != nullptr && mediaPaths != nullptr && mediaPaths.count > 0) {

        std::string marker = mtmd_default_marker();
        size_t pos = 0;
        while ((pos = prompt_str.find("<|image_pad|>", pos)) != std::string::npos) {
            prompt_str.replace(pos, 13, marker);
            pos += marker.length();
        }
        pos = 0;
        while ((pos = prompt_str.find("<|media_pad|>", pos)) != std::string::npos) {
            prompt_str.replace(pos, 13, marker);
            pos += marker.length();
        }

        std::vector<mtmd_bitmap *> bitmaps;
        for (NSString *mediaPath in mediaPaths) {
            mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(mtmd_ctx, [mediaPath UTF8String]);
            if (bmp) bitmaps.push_back(bmp);
        }

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text_input = { prompt_str.c_str(), true, true };

        int32_t tok_res = mtmd_tokenize(mtmd_ctx, chunks, &text_input, (const mtmd_bitmap **)bitmaps.data(), bitmaps.size());

        if (tok_res == 0) {
            llama_pos new_n_past = 0;

            // Only evaluates on Main Model context (Draft is skipped safely)
            mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, 0, 0, n_batch_size, true, &new_n_past);

            // --- CRITICAL FIX: Use the actual cursor position returned by the engine ---
            n_cur = new_n_past;
            n_prompt_tokens_total = n_cur;
        } else {
            // LOCAL PATCH: end the stream here instead of falling through.
            //
            // This used to log and set n_prompt_tokens_total = 0, then continue
            // into the generation loop below — where nothing had been decoded, so
            // llama_sampler_sample hit `get_logits_ith: invalid logits id -1,
            // reason: corrupt output buffer (n_outputs=0)` and llama.cpp called
            // ggml_abort. That is a SIGABRT, not a C++ exception: it kills the app
            // outright, and nl_sample_or_stop's try/catch cannot see it.
            //
            // mtmd_tokenize fails for ordinary, caller-reachable reasons, so this
            // is not a theoretical path. Chiefly: the number of media markers in
            // the prompt must equal the number of bitmaps (it returns 1 otherwise)
            // — and note a bitmap is silently skipped just above when
            // mtmd_helper_bitmap_init_from_file cannot read the file, so a corrupt
            // or unsupported image on its own is enough to arrive here with one
            // marker and zero bitmaps.
            //
            // Ending the stream matches how this file already handles a refused
            // generation and a throwing sampler: the caller gets an empty result
            // through its normal path instead of a dead process.
            NSLog(@"MTMD Tokenize failed with error code: %d "
                   "(media marker/bitmap mismatch, or an unreadable image). "
                   "Ending generation instead of sampling with no logits.", tok_res);
            mtmd_input_chunks_free(chunks);
            for (auto b : bitmaps) mtmd_bitmap_free(b);
            llama_sampler_free(smpl);
            // Same sentinel contract as the is_generating guard above: the Swift
            // layer drops a nil token and closes the Dart stream only on this
            // exact string, so returning without it would hang the caller.
            if (onToken) onToken(@"__END_OF_STREAM__");
            return;
        }

        mtmd_input_chunks_free(chunks);
        for (auto b : bitmaps) mtmd_bitmap_free(b);

    } else {
        // --- STANDARD TEXT-ONLY PATH (with KV prefix reuse) ---
        int n_prompt = -llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), NULL, 0, true, true);
        if (n_prompt < 0) n_prompt = -n_prompt;

        std::vector<llama_token> prompt_tokens(n_prompt);
        int tokenized_count = llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), prompt_tokens.data(), prompt_tokens.size(), true, true);
        prompt_tokens.resize(tokenized_count);

        n_prompt_tokens_total = prompt_tokens.size();

        // How much of this prompt is already sitting in the KV cache?
        //
        // A chat turn re-sends the whole conversation, so the new prompt is
        // almost always the previous one plus the model's last reply plus the
        // new user turn. Re-prefilling that shared head is the dominant cost of
        // time-to-first-token — thousands of tokens to recompute something the
        // cache already holds.
        const nl_prefix_plan plan =
            nl_plan_prefix_reuse(cached_tokens, prompt_tokens, may_reuse_prefix);
        size_t n_reuse = plan.n_reuse;
        const bool trim_wanted = plan.needs_trim;
        bool trim_refused = false;
        const double t_prefill_start = CFAbsoluteTimeGetCurrent();

        if (plan.needs_trim) {
            // The prompt diverged from the cache; evict the tail past the common
            // prefix. A false here means this architecture cannot be rewound
            // (see nl-kv-reuse.h) — fall back to a clean prefill, which is merely
            // slow instead of wrong.
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) n_reuse, -1)) {
                llama_memory_clear(llama_get_memory(ctx), true);
                n_reuse = 0;
                trim_refused = true;
            }
        } else if (plan.needs_clear) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }

        llama_batch batch = llama_batch_init(n_batch_size, 0, 1);
        int n_eval = (int) n_reuse;

        while (n_eval < (int)prompt_tokens.size()) {
            int n_chunk = MIN((int)prompt_tokens.size() - n_eval, n_batch_size);
            batch.n_tokens = 0;
            for (int i = 0; i < n_chunk; ++i) {
                batch.token[i] = prompt_tokens[n_eval + i];
                batch.pos[i] = n_eval + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (n_eval + i == prompt_tokens.size() - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx, batch) != 0) {
                // The cache no longer matches what we think it holds.
                [self invalidatePrefixCache];
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                return;
            }

            if (use_draft) {
                for (int i = 0; i < n_chunk; ++i) {
                    batch.logits[i] = (n_eval + i == prompt_tokens.size() - 1);
                }
                if (llama_decode(ctx_draft, batch) != 0) {
                    use_draft = false; // Fail safely
                }
            }
            n_eval += n_chunk;
        }
        n_cur = n_eval;
        llama_batch_free(batch);

        // The one number nobody could see. Everything about prefill cost was
        // guesswork from the outside — llama-server prints prompt_n/prompt_ms
        // and this did not, so on device there was no way to tell a working KV
        // reuse from a full re-prefill every turn. They differ by ~7x.
        //
        // Reads: "reused N of M, prefilled K in T ms". If `reused` stays near 0
        // across the turns of one scene, prefix reuse is not working and THAT is
        // the latency, not the model.
        // stderr, NOT NSLog. Everything llama.cpp prints goes to stderr, and
        // that is what gets captured when someone grabs a device log; NSLog
        // goes to os_log and does not appear there. The first version of this
        // line used NSLog and was simply absent from the log it was written for.
        fprintf(stderr,
                "nl_prefill: reused %zu of %zu, prefilled %d in %.0f ms (%s)%s\n",
                n_reuse, prompt_tokens.size(),
                (int)prompt_tokens.size() - (int)n_reuse,
                (CFAbsoluteTimeGetCurrent() - t_prefill_start) * 1000.0,
                trim_wanted ? "trim" : "append",
                trim_refused ? " TRIM REFUSED -> full prefill" : "");

        // The cache now holds exactly this prompt. Generated tokens are appended
        // as they are decoded, below. Only track it on the path that is allowed
        // to reuse it — with drafting on, accept/reject rollbacks move the cache
        // in ways this vector doesn't follow.
        if (may_reuse_prefix) cached_tokens = prompt_tokens;
    }
    // -----------------------------------------

    const uint32_t n_ctx_max = llama_n_ctx(ctx);
    const int n_draft = 5;
    bool is_eog_reached = false;

    llama_batch decode_batch = llama_batch_init(1, 0, 1);
    decode_batch.n_seq_id[0] = 1;
    decode_batch.seq_id[0][0] = 0;

    // --- CRITICAL FIX: Safe fallback for token count ---
    if (n_prompt_tokens_total <= 0) {
        n_prompt_tokens_total = n_cur;
    }

    bool sample_failed = false;

    // `llama_sampler_sample` applies the chain *and then accepts the token it
    // chose* into that chain — see the end of llama_sampler_sample() in
    // llama-sampler.cpp. So nothing below accepts again. Doing so advanced every
    // stateful sampler twice per token; for a grammar that meant its stack ran
    // two steps ahead of the text actually emitted, and the first token the
    // desynchronised stack could not accept threw std::runtime_error out of a
    // C++ path with no handler — SIGABRT, mid-generation.
    while (true) {
        if (stop_generation || is_eog_reached || sample_failed) break;

        if (n_cur + n_draft + 1 >= n_ctx_max) {
            int n_keep = n_prompt_tokens_total;
            if (n_keep >= n_ctx_max / 2) n_keep = n_ctx_max / 2;
            const int n_discard = (n_ctx_max - n_keep) / 2;

            // Context shift drops a window out of the middle of the sequence and
            // slides the tail back. On a recurrent/hybrid model that is exactly
            // the rewind seq_rm refuses to perform (see the prefill above), and
            // ignoring the refusal would leave the layer state describing tokens
            // that are no longer there — silent corruption for the rest of the
            // reply. Stop cleanly instead and let the caller's own truncation
            // policy deal with a conversation that outgrew the window.
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, n_keep + n_discard)) {
                NSLog(@"[LlamaBridge] Context full and this architecture cannot shift; ending generation");
                [self invalidatePrefixCache];
                break;
            }
            llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep + n_discard, n_cur, -n_discard);
            n_cur -= n_discard;

            // The shift renumbered positions; cached_tokens no longer describes
            // the cache, and a shifted context can't serve as a reusable prefix.
            [self invalidatePrefixCache];

            if (use_draft) {
                llama_memory_seq_rm(llama_get_memory(ctx_draft), 0, n_keep, n_keep + n_discard);
                llama_memory_seq_add(llama_get_memory(ctx_draft), 0, n_keep + n_discard, n_cur, -n_discard);
            }
        }

        std::vector<llama_token> draft_tokens;
        if (use_draft) {
            for (int i = 0; i < n_draft; ++i) {
                llama_token t = nl_sample_or_stop(smpl, ctx_draft, -1, &sample_failed);
                if (sample_failed) break;
                draft_tokens.push_back(t);

                decode_batch.token[0] = draft_tokens.back();
                decode_batch.pos[0] = n_cur + i;
                decode_batch.n_tokens = 1;
                decode_batch.logits[0] = true;

                if (llama_decode(ctx_draft, decode_batch) != 0) { break; }
            }
        }

        llama_token t_extra = nl_sample_or_stop(smpl, ctx, -1, &sample_failed);
        if (sample_failed) break;
        int n_accepted = 0;

        if (!draft_tokens.empty() && t_extra == draft_tokens[0]) {
            is_eog_reached = [self _sendToken:t_extra vocab:vocab onToken:onToken];
            n_accepted = 1;

            if (is_eog_reached || llama_vocab_is_eog(vocab, t_extra)) {
                is_eog_reached = true;
            } else {
                llama_batch b_tgt = llama_batch_init((int)draft_tokens.size(), 0, 1);
                for (int i = 0; i < (int)draft_tokens.size(); ++i) {
                    b_tgt.token[i] = draft_tokens[i];
                    b_tgt.pos[i] = n_cur + i;
                    b_tgt.n_seq_id[i] = 1;
                    b_tgt.seq_id[i][0] = 0;
                    b_tgt.logits[i] = true;
                }
                b_tgt.n_tokens = (int)draft_tokens.size();
                if (llama_decode(ctx, b_tgt) != 0) { llama_batch_free(b_tgt); break; }

                for (int i = 1; i < (int)draft_tokens.size(); ++i) {
                    llama_token t_verified = nl_sample_or_stop(smpl, ctx, i - 1, &sample_failed);
                    if (sample_failed) break;
                    if (t_verified == draft_tokens[i]) {
                        if ([self _sendToken:t_verified vocab:vocab onToken:onToken]) { is_eog_reached = true; }
                        n_accepted++;
                        if (is_eog_reached || llama_vocab_is_eog(vocab, t_verified)) { is_eog_reached = true; break; }
                    } else {
                        t_extra = t_verified;
                        break;
                    }
                }

                if (!is_eog_reached && !sample_failed && n_accepted == (int)draft_tokens.size()) {
                    t_extra = nl_sample_or_stop(smpl, ctx, (int)draft_tokens.size() - 1, &sample_failed);
                }

                if (n_accepted < (int)draft_tokens.size()) {
                    llama_memory_seq_rm(llama_get_memory(ctx), 0, n_cur + n_accepted, -1);
                    if (use_draft) llama_memory_seq_rm(llama_get_memory(ctx_draft), 0, n_cur + n_accepted, -1);
                }
                llama_batch_free(b_tgt);
            }
        }

        if (is_eog_reached || sample_failed) break;

        if ([self _sendToken:t_extra vocab:vocab onToken:onToken]) { is_eog_reached = true; }
        if (is_eog_reached || llama_vocab_is_eog(vocab, t_extra)) { is_eog_reached = true; break; }

        decode_batch.token[0] = t_extra;
        decode_batch.pos[0] = n_cur + n_accepted;
        decode_batch.n_tokens = 1;
        decode_batch.logits[0] = true;

        if (llama_decode(ctx, decode_batch) != 0) { [self invalidatePrefixCache]; break; }
        if (use_draft) llama_decode(ctx_draft, decode_batch);

        // Mirror the decode into the prefix cache. An end-of-generation token is
        // never decoded (the loop breaks above), so it correctly never lands here
        // — which is what keeps the cache a strict prefix of the NEXT prompt,
        // where the chat template supplies the turn terminator itself.
        if (may_reuse_prefix) cached_tokens.push_back(t_extra);

        n_cur += n_accepted + 1;
    }

    onToken(@"__END_OF_STREAM__");
    llama_batch_free(decode_batch);
    llama_sampler_free(smpl);
}

- (void)abortGeneration {
    stop_generation = true;
}

- (void)unload {
    stop_generation = true;
    while (is_generating) {
        [NSThread sleepForTimeInterval:0.01];
    }
    [self invalidatePrefixCache];
    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
    if (ctx_draft) { llama_free(ctx_draft); ctx_draft = nullptr; }
    if (model_draft) { llama_model_free(model_draft); model_draft = nullptr; }
    if (mtmd_ctx) { mtmd_free(mtmd_ctx); mtmd_ctx = nullptr; }
}

- (void)dispose {
    [self unload];
    llama_backend_free();
}

- (int)getCpuCores:(BOOL)performanceOnly {
    if (performanceOnly) {
        // iOS doesn't easily expose performance cores vs efficiency cores in a standard way
        // like Android's sysfs, but usually 2 or 4 is a safe bet for high performance cores.
        // We'll return active processor count as a fallback.
        int cores = (int)[[NSProcessInfo processInfo] activeProcessorCount];
        return cores > 2 ? cores / 2 : cores;
    }
    return (int)[[NSProcessInfo processInfo] activeProcessorCount];
}

- (BOOL)_sendToken:(llama_token)token vocab:(const struct llama_vocab *)vocab onToken:(void (^)(NSString *))onToken {
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n > 0) {
        std::string s(buf, n);
        if (s == "</s>" || s == "<|im_end|>" || s == "<|end|>") return YES;
        onToken([NSString stringWithUTF8String:s.c_str()]);
    }
    return NO;
}

@end