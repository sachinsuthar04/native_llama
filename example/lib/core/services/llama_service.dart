import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:native_llama/native_llama.dart';
import 'package:get/get.dart';

class ModelInfo {
  final String name;
  final String description;
  final String url;
  final String filename;
  final double sizeGB;
  final int minRamGB;

  ModelInfo({
    required this.name,
    required this.description,
    required this.url,
    required this.filename,
    required this.sizeGB,
    required this.minRamGB,
  });
}

class LlamaService {
  static final LlamaService instance = LlamaService._internal();
  LlamaService._internal();

  final NativeLlama plugin = NativeLlama();

  final isInitialized = false.obs;
  final isDraftInitialized = false.obs;
  final isVisionInitialized = false.obs;

  final List<ModelInfo> availableModels = [
    ModelInfo(
      name: "Qwen 2.5 1.5B (Q4_K_M)",
      description: "Fast, capable model for mobile.",
      url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
      filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
      sizeGB: 1.1,
      minRamGB: 3,
    ),
    ModelInfo(
      name: "SmolLM2 135M (Q8_0)",
      description: "Ultra-compact model for fast draft generation.",
      url: "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
      filename: "SmolLM2-135M-Instruct-Q8_0.gguf",
      sizeGB: 0.15,
      minRamGB: 1,
    ),
    ModelInfo(
      name: "Qwen2-VL 2B Instruct (Q4_K_M)",
      description: "Base Model for Vision Tasks.",
      url: "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
      filename: "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
      sizeGB: 1.6,
      minRamGB: 4,
    ),
    ModelInfo(
      name: "Qwen2-VL Vision Projector (MMPROJ)",
      description: "Required Projector for Qwen2-VL vision. Assign this in Settings.",
      url: "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
      filename: "mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
      sizeGB: 1.2,
      minRamGB: 4,
    ),
  ];

  Future<void> initModel(String path, {int? nCtx, int? nThreads, int? nGpuLayers}) async {
    await plugin.initModel(await _getTruePath(path), nCtx: nCtx, nThreads: nThreads, nGpuLayers: nGpuLayers);
    isInitialized.value = plugin.isInitialized;
  }

  Future<void> initDraftModel(String path, {int? nCtx, int? nThreads, int? nGpuLayers}) async {
    await plugin.initDraftModel(await _getTruePath(path), nCtx: nCtx, nThreads: nThreads, nGpuLayers: nGpuLayers);
    isDraftInitialized.value = plugin.isDraftInitialized;
  }

  Future<void> initVision(String path) async {
    await plugin.initVision(await _getTruePath(path));
    isVisionInitialized.value = plugin.isVisionInitialized;
  }

  Future<int> getCpuCores({bool performanceOnly = false}) async {
    return await plugin.getCpuCores(performanceOnly: performanceOnly);
  }

  Future<List<double>> getEmbedding(String text) async {
    return await plugin.getEmbedding(text);
  }

  Stream<String> generateResponse(
      List<Map<String, String>> messages, {
        List<String>? mediaPaths, // --- MODIFIED: Renamed to mediaPaths ---
        double temperature = 0.7,
        int topK = 40,
        double topP = 0.9,
      }) {
    return plugin.generateResponse(
      messages,
      mediaPaths: mediaPaths, // --- MODIFIED: Pass mediaPaths to plugin ---
      temperature: temperature,
      topK: topK,
      topP: topP,
    );
  }

  Future<void> abortGeneration() async {
    await plugin.abortGeneration();
  }

  void dispose() {
    plugin.dispose();
    isInitialized.value = false;
    isDraftInitialized.value = false;
    isVisionInitialized.value = false;
  }

  Future<String> _getTruePath(String path) async {
    if (path.isEmpty) return "";
    if (path.contains('/Documents/')) {
      final fileName = path.split('/').last;
      final docsDir = await getApplicationDocumentsDirectory();
      return "${docsDir.path}/models/$fileName";
    }
    return path;
  }

  Future<String> getLocalModelsPath() async {
    final directory = await getApplicationDocumentsDirectory();
    final path = "${directory.path}/models";
    final dir = Directory(path);
    if (!await dir.exists()) await dir.create(recursive: true);
    return path;
  }

  Future<List<File>> getDownloadedModels() async {
    final path = await getLocalModelsPath();
    final dir = Directory(path);
    if (!await dir.exists()) return [];
    return dir.listSync().whereType<File>().where((file) => file.path.endsWith(".gguf")).toList();
  }

  Future<String> importModel(String sourcePath) async {
    final destinationDir = await getLocalModelsPath();
    final fileName = sourcePath.split('/').last;
    final destinationPath = "$destinationDir/$fileName";
    await File(sourcePath).copy(destinationPath);
    return destinationPath;
  }
}