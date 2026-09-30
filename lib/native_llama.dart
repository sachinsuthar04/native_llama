import 'dart:async';
import 'package:flutter/services.dart';

class NativeLlama {
  static const MethodChannel _methodChannel = MethodChannel('native_llama/methods');
  static const EventChannel _eventChannel = EventChannel('native_llama/events');

  bool _isInitialized = false;
  bool _isDraftInitialized = false;
  bool _isVisionInitialized = false; // Note: Used for Multimodal Projector (mmproj)

  bool get isInitialized => _isInitialized;
  bool get isDraftInitialized => _isDraftInitialized;
  bool get isVisionInitialized => _isVisionInitialized;

  /// Initializes the main base model
  /// [nCtx] Optional override for context window. If null, calculates based on device RAM.
  /// [nThreads] Optional override for CPU threads. Defaults to 4.
  /// [nGpuLayers] Number of layers to offload to GPU. -1 for auto (all), 0 for CPU only.
  Future<void> initModel(String absolutePath, {int? nCtx, int? nThreads, int? nGpuLayers}) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initModel', {
        'modelPath': absolutePath,
        'nCtx': nCtx,
        'nThreads': nThreads,
        'nGpuLayers': nGpuLayers ?? 0,
      });
      _isInitialized = result;
      if (!result) throw Exception("Native initialization failed. The model may be too large for this device's memory.");
    } on PlatformException catch (e) {
      _isInitialized = false;
      throw Exception("Platform Exception during init: ${e.message}");
    }
  }

  /// Initializes the draft model for speculative decoding
  Future<void> initDraftModel(String absolutePath, {int? nCtx, int? nThreads, int? nGpuLayers}) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initDraftModel', {
        'modelPath': absolutePath,
        'nCtx': nCtx,
        'nThreads': nThreads,
        'nGpuLayers': nGpuLayers ?? 0,
      });
      _isDraftInitialized = result;
      if (!result) throw Exception("Native initialization failed for draft model.");
    } on PlatformException catch (e) {
      _isDraftInitialized = false;
      throw Exception("Platform Exception during draft init: ${e.message}");
    }
  }

  /// Initialize the Multimodal Projector (MTMD)
  Future<void> initVision(String absolutePath) async {
    try {
      final bool result = await _methodChannel.invokeMethod('initVision', {
        'mmprojPath': absolutePath,
      });
      _isVisionInitialized = result;
      if (!result) throw Exception("Native initialization failed for multimodal projector.");
    } on PlatformException catch (e) {
      _isVisionInitialized = false;
      throw Exception("Platform Exception during vision init: ${e.message}");
    }
  }

  /// Extracts embeddings for Vector DB/RAG integration
  Future<List<double>> getEmbedding(String text) async {
    if (!_isInitialized) throw Exception("Model not initialized.");
    try {
      final List<dynamic>? result = await _methodChannel.invokeMethod('getEmbedding', {
        'text': text,
      });
      return result?.cast<double>() ?? [];
    } catch (e) {
      print("Embedding Error: $e");
      return [];
    }
  }

  /// Generates response and streams tokens back to the UI
  /// [repeatPenalty] / [penaltyLastN] / [freqPenalty] / [presencePenalty] control
  /// the sampler's repetition penalty. The defaults preserve the long-standing
  /// baked-in values. Pass `repeatPenalty: 1.0, freqPenalty: 0, presencePenalty: 0`
  /// to turn it OFF for structured/list output (e.g. a JSON array of similar
  /// objects), which the penalty otherwise truncates by penalising the repeated
  /// structural tokens.
  ///
  /// ## KV prefix reuse — why message ORDER affects latency
  ///
  /// The backend remembers the exact token sequence left in the KV cache and, on
  /// the next call, re-prefills only the part of the prompt that actually
  /// changed. In a normal chat — where each turn re-sends the whole conversation
  /// plus one more exchange — that turns a multi-thousand-token prefill into a
  /// few dozen tokens, which is most of the time-to-first-token.
  ///
  /// It happens automatically; there is nothing to enable. But it only pays off
  /// to the extent that the *start* of the prompt is unchanged, so put the
  /// stable content first:
  ///
  /// ```
  ///   system rules → persona → world state → retrieved context → turns → ask
  ///   └────────────── stable, reused ──────────────┘ └── the part that changes ──┘
  /// ```
  ///
  /// Editing an early message (rewriting the system prompt, swapping retrieved
  /// context that sits ahead of the dialogue) invalidates everything after it and
  /// costs a full prefill. Where content changes per turn, append it to the last
  /// user message instead of splicing it into a system block up front.
  ///
  /// Reuse is skipped, and the context starts clean, when:
  ///  * [mediaPaths] is non-empty — the vision path owns its own position cursor;
  ///  * a draft model is loaded for speculative decoding (note that passing a
  ///    [grammar] already disables drafting, so grammared calls keep reuse);
  ///  * [resetContext] or [dispose] has run since the last generation.
  ///
  /// Reuse is never *silently* partial: if the architecture cannot rewind its
  /// cache — which is the case for recurrent/hybrid models such as LFM2, whose
  /// shortconv layers hold a rolling window rather than per-position state — the
  /// backend falls back to a full, correct prefill. The observable behaviour of
  /// this method is identical either way; only its speed changes.
  Stream<String> generateResponse(
      List<Map<String, String>> messages, {
        List<String>? mediaPaths, // --- MODIFIED: Accepts images & audio files ---
        double temperature = 0.7,
        int topK = 40,
        double topP = 0.9,
        double repeatPenalty = 1.2,
        int penaltyLastN = 128,
        double freqPenalty = 0.1,
        double presencePenalty = 0.1,
        /// Optional GBNF grammar constraining the output.
        ///
        /// When supplied, the backend masks every token the grammar forbids
        /// before temperature/top-k/top-p are applied, so structurally invalid
        /// output becomes impossible rather than merely unlikely — worth far
        /// more than prompt instructions for JSON or fixed-format replies. The
        /// start symbol must be named `root`. A grammar that fails to parse is
        /// ignored and generation continues unconstrained.
        String? grammar,
      }) {
    if (!_isInitialized) {
      return Stream.error("Model not initialized");
    }

    final StreamController<String> controller = StreamController<String>();
    StreamSubscription? subscription;

    controller.onListen = () {
      subscription = _eventChannel.receiveBroadcastStream().listen(
            (event) {
          final token = event.toString();

          if (token == "__END_OF_STREAM__") {
            if (!controller.isClosed) controller.close();
          } else {
            controller.add(token);
          }
        },
        onDone: () {
          if (!controller.isClosed) controller.close();
        },
        onError: (e) {
          if (!controller.isClosed) controller.addError(e);
        },
      );

      controller.onCancel = () {
        abortGeneration();
        subscription?.cancel();
      };

      final roles = messages.map((m) => m['role'] ?? 'user').toList();
      final contents = messages.map((m) => m['content'] ?? m['text'] ?? '').toList();

      _methodChannel.invokeMethod('startGeneration', {
        'roles': roles,
        'contents': contents,
        'mediaPaths': mediaPaths ?? [], // --- MODIFIED: Pass media array to Native ---
        'temperature': temperature,
        'topK': topK,
        'topP': topP,
        'repeatPenalty': repeatPenalty,
        'penaltyLastN': penaltyLastN,
        'freqPenalty': freqPenalty,
        'presencePenalty': presencePenalty,
        'grammar': grammar,
      }).catchError((e) {
        if (!controller.isClosed) controller.addError(e);
      });
    };

    return controller.stream;
  }

  /// Force stops the current generation loop
  Future<void> abortGeneration() async {
    await _methodChannel.invokeMethod('abortGeneration');
  }

  /// Resets the generation context (KV cache + positions) WITHOUT unloading the
  /// model weights or the vision projector, so the next generation starts from a
  /// clean state at a fraction of a full reload's cost. This is the safe way to
  /// process several images in sequence: a full reset per image keeps each one's
  /// M-RoPE positions clean without paying the multi-GB weight reload each time.
  ///
  /// Returns true on success, false when the model isn't loaded or the running
  /// native build predates this method (older builds report it as not
  /// implemented). On false the caller should fall back to dispose + re-init.
  ///
  /// [nCtx] should match the value passed to [initModel]; the context is
  /// recreated with the same window.
  Future<bool> resetContext({int? nCtx}) async {
    if (!_isInitialized) return false;
    try {
      final bool ok = await _methodChannel.invokeMethod('resetContext', {
        'nCtx': nCtx,
      });
      return ok;
    } on MissingPluginException {
      return false; // native build without resetContext — caller falls back
    } on PlatformException {
      return false;
    }
  }

  /// Safely disposes of models to prevent OOM crashes
  Future<void> dispose() async {
    try {
      await _methodChannel.invokeMethod('dispose');
      _isInitialized = false;
      _isDraftInitialized = false;
      _isVisionInitialized = false;
    } catch (e) {
      print("Error disposing model: $e");
    }
  }

  /// Gets the number of CPU cores.
  /// [performanceOnly] If true, attempts to return only high-performance cores (Android only).
  Future<int> getCpuCores({bool performanceOnly = false}) async {
    try {
      return await _methodChannel.invokeMethod('getCpuCores', {
        'performanceOnly': performanceOnly,
      }) ?? 4;
    } catch (e) {
      return 4;
    }
  }
}