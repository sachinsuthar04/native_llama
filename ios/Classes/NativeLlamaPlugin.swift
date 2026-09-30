import Flutter
import UIKit

public class NativeLlamaPlugin: NSObject, FlutterPlugin, FlutterStreamHandler {
    private var eventSink: FlutterEventSink?

    public static func register(with registrar: FlutterPluginRegistrar) {
        let methodChannel = FlutterMethodChannel(name: "native_llama/methods", binaryMessenger: registrar.messenger())
        let eventChannel = FlutterEventChannel(name: "native_llama/events", binaryMessenger: registrar.messenger())

        let instance = NativeLlamaPlugin()
        registrar.addMethodCallDelegate(instance, channel: methodChannel)
        eventChannel.setStreamHandler(instance)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initModel":
            guard let args = call.arguments as? [String: Any],
                  let modelPath = args["modelPath"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Model path is null", details: nil))
                return
            }

            let nCtx = args["nCtx"] as? Int ?? -1
            let nThreads = args["nThreads"] as? Int ?? -1
            let nGpuLayers = args["nGpuLayers"] as? Int ?? 99

            // Move heavy initialization to a background thread
            DispatchQueue.global(qos: .userInitiated).async {
                let success = LlamaBridge.shared().initModel(modelPath, nCtx: Int32(nCtx), nThreads: Int32(nThreads), nGpuLayers: Int32(nGpuLayers))
                DispatchQueue.main.async { result(success) }
            }

        case "initDraftModel":
            guard let args = call.arguments as? [String: Any],
                  let modelPath = args["modelPath"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Model path is null", details: nil))
                return
            }

            let nCtx = args["nCtx"] as? Int ?? -1
            let nThreads = args["nThreads"] as? Int ?? -1
            let nGpuLayers = args["nGpuLayers"] as? Int ?? 99

            DispatchQueue.global(qos: .userInitiated).async {
                let success = LlamaBridge.shared().initDraftModel(modelPath, nCtx: Int32(nCtx), nThreads: Int32(nThreads), nGpuLayers: Int32(nGpuLayers))
                DispatchQueue.main.async { result(success) }
            }

        case "initVision":
            guard let args = call.arguments as? [String: Any],
                  let mmprojPath = args["mmprojPath"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Vision model path is null", details: nil))
                return
            }

            DispatchQueue.global(qos: .userInitiated).async {
                let success = LlamaBridge.shared().initVision(mmprojPath)
                DispatchQueue.main.async { result(success) }
            }

        case "getEmbedding":
            guard let args = call.arguments as? [String: Any],
                  let text = args["text"] as? String else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Text is null", details: nil))
                return
            }
            DispatchQueue.global(qos: .userInitiated).async {
                let embedding = LlamaBridge.shared().getEmbedding(text)
                DispatchQueue.main.async { result(embedding) }
            }

        case "startGeneration":
            guard let args = call.arguments as? [String: Any],
                  let roles = args["roles"] as? [String],
                  let contents = args["contents"] as? [String] else {
                result(FlutterError(code: "INVALID_ARGUMENT", message: "Roles or contents are null", details: nil))
                return
            }

            // --- MODIFIED: Extract media paths instead of image paths ---
            let mediaPaths = args["mediaPaths"] as? [String] ?? []

            let temperature = (args["temperature"] as? NSNumber)?.floatValue ?? 0.7
            let topK = args["topK"] as? Int ?? 40
            let topP = (args["topP"] as? NSNumber)?.floatValue ?? 0.9
            // Repetition penalty (defaults preserve the previous baked-in values).
            let repeatPenalty = (args["repeatPenalty"] as? NSNumber)?.floatValue ?? 1.2
            let penaltyLastN = args["penaltyLastN"] as? Int ?? 128
            let freqPenalty = (args["freqPenalty"] as? NSNumber)?.floatValue ?? 0.1
            let presencePenalty = (args["presencePenalty"] as? NSNumber)?.floatValue ?? 0.1
            // Optional GBNF grammar; nil means unconstrained sampling.
            let grammar = args["grammar"] as? String

            // Prevent screen from sleeping during long generations
            UIApplication.shared.isIdleTimerDisabled = true

            DispatchQueue.global(qos: .userInitiated).async {
                // Pass mediaPaths to the bridge
                LlamaBridge.shared().startGeneration(withRoles: roles,
                                                     contents: contents,
                                                     mediaPaths: mediaPaths.isEmpty ? nil : mediaPaths,
                                                     temperature: temperature,
                                                     topK: Int32(topK),
                                                     topP: topP,
                                                     repeatPenalty: repeatPenalty,
                                                     penaltyLastN: Int32(penaltyLastN),
                                                     freqPenalty: freqPenalty,
                                                     presencePenalty: presencePenalty,
                                                     grammar: grammar) { [weak self] token in
                    guard let token = token else { return }
                    DispatchQueue.main.async {
                        if token == "__END_OF_STREAM__" {
                            UIApplication.shared.isIdleTimerDisabled = false
                            self?.eventSink?(token)
                            self?.eventSink?(FlutterEndOfEventStream)
                        } else {
                            self?.eventSink?(token)
                        }
                    }
                }
            }
            result(nil)

        case "abortGeneration":
            LlamaBridge.shared().abortGeneration()
            result(true)

        case "resetContext":
            let nCtx = (call.arguments as? [String: Any])?["nCtx"] as? Int ?? -1
            DispatchQueue.global(qos: .userInitiated).async {
                let success = LlamaBridge.shared().resetContext(Int32(nCtx))
                DispatchQueue.main.async { result(success) }
            }

        case "dispose":
            DispatchQueue.global(qos: .userInitiated).async {
                LlamaBridge.shared().dispose()
                DispatchQueue.main.async { result(true) }
            }

        case "getCpuCores":
            let performanceOnly = (call.arguments as? [String: Any])?["performanceOnly"] as? Bool ?? false
            let cores = LlamaBridge.shared().getCpuCores(performanceOnly)
            result(Int(cores))

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - FlutterStreamHandler
    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        return nil
    }
}