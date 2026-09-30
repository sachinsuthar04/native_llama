# native_llama

A high-performance Flutter plugin for running **Llama** and other LLMs locally on mobile devices. Powered by `llama.cpp` and `MTMD`, it supports Android (Vulkan) and iOS (Metal) acceleration.

## Features

- 🚀 **Local Inference**: Run LLMs entirely on-device (no internet required).
- ⚡ **Hardware Acceleration**: 
  - **Android**: Vulkan support for high-performance GPU offloading.
  - **iOS**: Metal support optimized for Apple Silicon.
- 👁️ **Multimodal (Vision)**: Support for vision models (e.g., Qwen2-VL) via multimodal projectors (`mmproj`).
- ⏩ **Speculative Decoding**: Use a smaller "draft" model to accelerate generation from a larger "target" model.
- 🧠 **Embeddings**: Extract high-quality text embeddings for RAG and semantic search.
- 🛠️ **Fine-grained Control**:
  - Control CPU thread count (auto-detects performance cores).
  - Adjust GPU layer offloading.
  - Dynamic context window management.

## Installation

Add `native_llama` to your `pubspec.yaml`:

```yaml
dependencies:
  native_llama: ^1.0.0
```

### Android Setup
Ensure your `minSdkVersion` is at least **28** in `android/app/build.gradle`.
For optimal performance, add `android:largeHeap="true"` and `android:extractNativeLibs="true"` to your `AndroidManifest.xml`.

### iOS Setup
Ensure your deployment target is at least **iOS 13.0**.
Metal acceleration is enabled by default on compatible devices.

## Usage

### Initialize Model
```dart
final llama = NativeLlama();

await llama.initModel(
  "/path/to/model.gguf",
  nCtx: 2048,
  nThreads: 4,
  nGpuLayers: -1, // -1 for full GPU offload
);
```

### Text Generation (Streaming)
```dart
final messages = [
  {'role': 'system', 'text': 'You are a helpful assistant.'},
  {'role': 'user', 'text': 'Hello!'},
];

llama.generateResponse(messages).listen((token) {
  print("Received token: $token");
});
```

### Multimodal Vision
```dart
await llama.initVision("/path/to/mmproj.gguf");

llama.generateResponse(
  messages,
  mediaPaths: ["/path/to/image.jpg"],
).listen((token) {
  // ...
});
```

### KV Prefix Reuse (automatic)

Repeated `generateResponse` calls re-prefill only what changed. A chat turn
re-sends the whole conversation, so without this every turn recomputes thousands
of tokens the KV cache already holds — which is most of the time-to-first-token.

Nothing to enable. To benefit, order the prompt **most-stable-content-first**:

```
system rules → persona → world state → retrieved context → turns → the ask
└───────────────── reused across turns ─────────────────┘ └── re-prefilled ──┘
```

Changing an early message invalidates everything after it. Where content varies
per turn, append it to the last user message rather than splicing it into a
system block up front.

Reuse is skipped (the context starts clean) for vision calls, when a draft model
is loaded, and after `resetContext()` / `dispose()`. Note that passing a
`grammar` already disables drafting, so grammared calls keep reuse.

**Correctness on recurrent/hybrid models.** Architectures such as Mamba, RWKV and
**LFM2** hold a rolling window per layer rather than per-position state, so their
cache cannot be rewound — `llama_memory_recurrent::seq_rm` refuses a partial
truncation rather than corrupt itself. Only *pure append* (the cache is a strict
prefix of the new prompt) is reused there; anything divergent falls back to a
full, correct prefill. Behaviour is identical either way — only speed changes.

The planner lives in `ios/shared_cpp/common/nl-kv-reuse.h`, is shared by both
platforms, and is covered by `test/nl_kv_reuse_test.cpp`:

```bash
c++ -std=c++17 -I ios/shared_cpp/include -I ios/shared_cpp/ggml/include \
    -I ios/shared_cpp/common test/nl_kv_reuse_test.cpp -o /tmp/t && /tmp/t
```

### Speculative Decoding
```dart
await llama.initDraftModel("/path/to/small_draft_model.gguf");

// Generation will automatically use the draft model for speedup
llama.generateResponse(messages).listen((token) => ...);
```

## Bonsai 1-bit Models (Q1_0)

`native_llama` supports PrismML's [**Bonsai**](https://huggingface.co/collections/prism-ml/bonsai) 1-bit LLMs in the `Q1_0_g128` GGUF format (ggml tensor type 41: weights quantized to {-1, +1} with FP16 scales per 128-weight group). These models trade a small amount of quality for dramatically smaller downloads and memory footprints — ideal for on-device use.

| Model | File | Size |
|-------|------|------|
| Bonsai-1.7B | `Bonsai-1.7B-Q1_0.gguf` | ~237 MB |
| Bonsai-4B   | `Bonsai-4B-Q1_0.gguf`   | ~546 MB |
| Bonsai-8B   | `Bonsai-8B-Q1_0.gguf`   | ~1.08 GB |

No special API is needed — load the `.gguf` like any other model:

```dart
await llama.initModel("/path/to/Bonsai-1.7B-Q1_0.gguf");
```

**How it runs:** Q1_0 is executed by optimized kernels on every backend —
Metal on iOS (`kernel_mul_mv_q1_0_f32`), Vulkan on Android (native `q1_0`
matmul/dequant pipelines), and ARM NEON on CPU (i8mm/DOTPROD/plain-NEON paths,
ported from the PrismML fork). On devices without Vulkan support, Q1_0 tensors
fall back to the CPU backend automatically via llama.cpp's backend scheduler.

**Performance notes** (Bonsai-1.7B-Q1_0, measured on an Apple-silicon Mac,
greedy decode): ~101 tok/s with Metal offload, ~50 tok/s CPU-only. Expect
lower numbers on mobile SoCs, with the same Metal/Vulkan speedup pattern.

**Compatibility:** the fork-only `Q2_0` 2-bit format (type 42) used by some
PrismML releases is **not** supported by the vendored llama.cpp snapshot —
use the `Q1_0` files. Bonsai-27B (Qwen3.6 hybrid architecture) is untested.

**Attribution:** Bonsai model weights are © PrismML, licensed under
[Apache-2.0](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/blob/main/LICENSE).
The Q1_0 ggml kernels are part of llama.cpp (MIT, © the ggml authors); the ARM
NEON Q1_0 dot-product optimizations are ported from
[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (MIT).

## Example App
Check the `example` directory for a full-featured chat application that includes:
- Model downloading and management.
- Hardware configuration settings.
- Image attachment support.
- Real-time performance metrics.

## License
MIT
