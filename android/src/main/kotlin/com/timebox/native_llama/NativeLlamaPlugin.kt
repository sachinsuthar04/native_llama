package com.timebox.native_llama

import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.annotation.Keep
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.util.concurrent.Executors

class NativeLlamaPlugin: FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler {
    private lateinit var methodChannel: MethodChannel
    private lateinit var eventChannel: EventChannel

    private var eventSink: EventChannel.EventSink? = null
    private val handler = Handler(Looper.getMainLooper())
    // Executes C++ inference on a background thread to keep Flutter UI smooth
    private val executor = Executors.newSingleThreadExecutor()

    companion object {
        private const val TAG = "NativeLlamaPlugin"

        /// Whether libnative_llama.so is present and loadable on this device.
        ///
        /// The library is built for arm64-v8a only (see `abiFilters` in
        /// android/build.gradle.kts), so on any other ABI the load fails and
        /// every `external` method below would throw UnsatisfiedLinkError.
        ///
        /// This used to be a bare `System.loadLibrary` in a companion `init`,
        /// which made that an Error thrown from the *class initializer* — i.e.
        /// from `new NativeLlamaPlugin()` inside GeneratedPluginRegistrant.
        /// That registrant only catches `Exception`, and UnsatisfiedLinkError
        /// is an `Error`, so it escaped configureFlutterEngine() and killed the
        /// process before Dart main() ever ran. Google Play review (x86_64
        /// emulators) saw that as "crashes after opening", and Crashlytics
        /// never reported it because a 100%-reproducible startup crash can
        /// never upload the report it queued on the previous launch.
        ///
        /// Catching Throwable here degrades the LLM features to unavailable
        /// instead of taking the whole app down with them.
        @JvmStatic
        val isAvailable: Boolean = try {
            System.loadLibrary("native_llama")
            true
        } catch (t: Throwable) {
            Log.e(
                TAG,
                "libnative_llama.so unavailable on ABI " +
                    "${Build.SUPPORTED_ABIS.firstOrNull()} — LLM features disabled",
                t
            )
            false
        }
    }

    // Native JNI bindings
    private external fun initLlama(modelPath: String, nCtx: Int, nThreads: Int, nGpuLayers: Int): Boolean
    private external fun initDraftModel(modelPath: String, nCtx: Int, nThreads: Int, nGpuLayers: Int): Boolean

    // --- Vision/Media Model Call ---
    private external fun initVision(mmprojPath: String): Boolean

    private external fun getCpuCores(performanceOnly: Boolean): Int

    private external fun getEmbedding(text: String): DoubleArray?

    // --- MODIFIED: Renamed imagePaths to mediaPaths ---
    private external fun startNativeGeneration(roles: Array<String>, contents: Array<String>, mediaPaths: Array<String>, temperature: Float, topK: Int, topP: Float, repeatPenalty: Float, penaltyLastN: Int, freqPenalty: Float, presencePenalty: Float, grammar: String?)

    private external fun abortGeneration()

    // Recreates only the llama context (KV cache + positions), keeping weights +
    // vision projector resident — a cheap, clean reset between images.
    private external fun resetContext(nCtx: Int): Boolean

    private external fun disposeLlama()

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel = MethodChannel(flutterPluginBinding.binaryMessenger, "native_llama/methods")
        methodChannel.setMethodCallHandler(this)

        eventChannel = EventChannel(flutterPluginBinding.binaryMessenger, "native_llama/events")
        eventChannel.setStreamHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        // Without the native library every branch below would throw
        // UnsatisfiedLinkError off the executor thread, where nothing catches
        // it. Fail the call instead so Dart can fall back to a non-LLM path.
        if (!isAvailable) {
            result.error(
                "NATIVE_UNAVAILABLE",
                "native_llama is not available on this device's ABI " +
                    "(${Build.SUPPORTED_ABIS.firstOrNull()}).",
                null
            )
            return
        }
        when (call.method) {
            "initModel" -> {
                val modelPath = call.argument<String>("modelPath")
                val nCtx = call.argument<Int>("nCtx") ?: -1
                val nThreads = call.argument<Int>("nThreads") ?: -1
                val nGpuLayers = call.argument<Int>("nGpuLayers") ?: 0

                if (modelPath != null) {
                    executor.execute {
                        val success = initLlama(modelPath, nCtx, nThreads, nGpuLayers)
                        handler.post { result.success(success) }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Model path is null", null)
                }
            }
            "initDraftModel" -> {
                val modelPath = call.argument<String>("modelPath")
                val nCtx = call.argument<Int>("nCtx") ?: -1
                val nThreads = call.argument<Int>("nThreads") ?: -1
                val nGpuLayers = call.argument<Int>("nGpuLayers") ?: 0

                if (modelPath != null) {
                    executor.execute {
                        val success = initDraftModel(modelPath, nCtx, nThreads, nGpuLayers)
                        handler.post { result.success(success) }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Draft model path is null", null)
                }
            }
            "initVision" -> {
                val mmprojPath = call.argument<String>("mmprojPath")
                if (mmprojPath != null) {
                    executor.execute {
                        val success = initVision(mmprojPath)
                        handler.post { result.success(success) }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Vision model path is null", null)
                }
            }
            "getEmbedding" -> {
                val text = call.argument<String>("text")
                if (text != null) {
                    executor.execute {
                        val embedding = getEmbedding(text)
                        handler.post {
                            if (embedding != null) result.success(embedding.toList())
                            else result.error("EMBEDDING_ERROR", "Failed to get embedding", null)
                        }
                    }
                } else {
                    result.error("INVALID_ARGUMENT", "Text is null", null)
                }
            }
            "startGeneration" -> {
                val roles = call.argument<List<String>>("roles")?.toTypedArray()
                val contents = call.argument<List<String>>("contents")?.toTypedArray()

                // --- MODIFIED: Extract mediaPaths (defaults to empty array if none) ---
                val mediaPaths = call.argument<List<String>>("mediaPaths")?.toTypedArray() ?: emptyArray()

                // Cast Double from Dart to Float for Kotlin/C++ boundary
                val temperature = call.argument<Double>("temperature")?.toFloat() ?: 0.7f
                val topK = call.argument<Int>("topK") ?: 40
                val topP = call.argument<Double>("topP")?.toFloat() ?: 0.9f
                // Repetition penalty (defaults preserve the previous baked-in values).
                val repeatPenalty = call.argument<Double>("repeatPenalty")?.toFloat() ?: 1.2f
                val penaltyLastN = call.argument<Int>("penaltyLastN") ?: 128
                val freqPenalty = call.argument<Double>("freqPenalty")?.toFloat() ?: 0.1f
                val presencePenalty = call.argument<Double>("presencePenalty")?.toFloat() ?: 0.1f
                // Optional GBNF grammar; null means unconstrained sampling.
                val grammar = call.argument<String>("grammar")

                if (roles != null && contents != null) {
                    executor.execute {
                        startNativeGeneration(roles, contents, mediaPaths, temperature, topK, topP, repeatPenalty, penaltyLastN, freqPenalty, presencePenalty, grammar)
                    }
                    result.success(null)
                } else {
                    result.error("INVALID_ARGUMENT", "Roles or contents are null", null)
                }
            }
            "abortGeneration" -> {
                abortGeneration()
                result.success(true)
            }
            "resetContext" -> {
                val nCtx = call.argument<Int>("nCtx") ?: -1
                executor.execute {
                    val success = resetContext(nCtx)
                    handler.post { result.success(success) }
                }
            }
            "dispose" -> {
                executor.execute {
                    disposeLlama()
                    handler.post { result.success(true) }
                }
            }
            "getCpuCores" -> {
                val performanceOnly = call.argument<Boolean>("performanceOnly") ?: false
                executor.execute {
                    val cores = getCpuCores(performanceOnly)
                    handler.post { result.success(cores) }
                }
            }
            else -> result.notImplemented()
        }
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
    }

    override fun onCancel(arguments: Any?) {
        eventSink = null
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        methodChannel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        // Guarded for the same reason as onMethodCall: with no native library
        // this is an UnsatisfiedLinkError on the way *out* of the engine, which
        // would crash teardown on every non-arm64 device.
        //
        // Posted to the executor, NOT run here. onDetachedFromEngine arrives on
        // the Android main thread, and disposing the model frees ~1.25 GB of
        // weights and tears down a GPU (Vulkan/Metal) context that may still
        // have work in flight — seconds of blocking, on the one thread that
        // must never block. Held on the main thread it is an ANR while the user
        // is leaving the app, which is the worst place to be reported from
        // because nothing else about the app looks wrong.
        //
        // The executor is single-threaded, so this is also correctly ordered
        // behind any generation still queued on it: free-after-use, never
        // during. Nothing reads the engine after detach, so there is no result
        // to wait for.
        if (isAvailable) {
            executor.execute {
                try {
                    disposeLlama()
                } catch (t: Throwable) {
                    // Teardown of a process that is going away regardless.
                }
            }
        }
    }

    // FIX: Push EOS string explicitly so Dart can catch it manually
    @Keep
    fun onTokenReceived(token: String) {
        handler.post {
            eventSink?.success(token)
            if (token == "__END_OF_STREAM__") {
                eventSink?.endOfStream()
            }
        }
    }
}