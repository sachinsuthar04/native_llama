import 'dart:io';
import 'dart:isolate';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_downloader/flutter_downloader.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:file_picker/file_picker.dart';
import '../../../core/services/llama_service.dart';

class SettingsController extends GetxController {
  final LlamaService _llamaService = LlamaService.instance;

  var downloadedModels = <File>[].obs;
  var selectedModelPath = "".obs;
  var selectedDraftModelPath = "".obs;
  var selectedVisionModelPath = "".obs; // --- NEW: Vision Model Path ---

  var isModelLoaded = false.obs;
  var isDraftModelLoaded = false.obs;
  var isVisionModelLoaded = false.obs;  // --- NEW: Vision Loaded State ---

  var isDownloading = false.obs;
  var downloadProgress = 0.0.obs;
  var currentDownloadingModel = "".obs;

  var nThreads = 4.obs;
  var nCtx = 0.obs;
  var nGpuLayers = 0.obs; // 0 for CPU, > 0 for partial GPU, -1 for full GPU
  var useGpu = false.obs;

  var maxCores = 8.obs;
  var performanceCores = 4.obs;

  // --- MODIFIED: Filter out mmproj files from main text models ---
  List<File> get mainModels => downloadedModels
      .where((f) => f.existsSync() && f.lengthSync() > 1024 * 1024 * 500 && !f.path.toLowerCase().contains('mmproj'))
      .toList();

  List<File> get draftModels => downloadedModels
      .where((f) => f.existsSync() && f.lengthSync() <= 1024 * 1024 * 500 && !f.path.toLowerCase().contains('mmproj'))
      .toList();

  // --- NEW: Vision Models Getter ---
  List<File> get visionModels => downloadedModels
      .where((f) => f.existsSync() && f.path.toLowerCase().contains('mmproj'))
      .toList();

  final ReceivePort _port = ReceivePort();
  String? _activeTaskId;

  @override
  void onInit() {
    super.onInit();
    _loadSettings();
    _setupDownloadListener();
    _checkRunningTasks();
    _detectCores();
  }

  Future<void> _detectCores() async {
    maxCores.value = await _llamaService.getCpuCores(performanceOnly: false);
    performanceCores.value = await _llamaService.getCpuCores(performanceOnly: true);

    // If threads not set, default to performance cores
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getInt('nThreads') == null) {
      nThreads.value = performanceCores.value;
    }
  }

  Future<void> importModelFile() async {
    try {
      FilePickerResult? result = await FilePicker.pickFiles(type: FileType.any);
      if (result != null && result.files.single.path != null) {
        final path = result.files.single.path!;
        if (!path.endsWith('.gguf')) {
          Get.snackbar("Invalid File", "Please select a .gguf model file.");
          return;
        }

        Get.snackbar("Importing", "Copying model to app storage...");
        final newPath = await _llamaService.importModel(path);
        await _refreshDownloadedModels();
      }
    } catch (e) {
      Get.snackbar("Error", "Failed to import model: $e");
    }
  }

  @override
  void onClose() {
    IsolateNameServer.removePortNameMapping('downloader_send_port');
    super.onClose();
  }

  void _setupDownloadListener() {
    IsolateNameServer.removePortNameMapping('downloader_send_port');
    IsolateNameServer.registerPortWithName(_port.sendPort, 'downloader_send_port');

    _port.listen((dynamic data) {
      String id = data[0];
      int statusValue = data[1];
      int progress = data[2];

      if (_activeTaskId == id || _activeTaskId == null) {
        if (progress > 0) downloadProgress.value = progress / 100;

        final status = DownloadTaskStatus.fromInt(statusValue);
        if (status == DownloadTaskStatus.complete) {
          isDownloading.value = false;
          _activeTaskId = null;
          Get.snackbar("Success", "Model downloaded successfully");
          _refreshDownloadedModels();
        } else if (status == DownloadTaskStatus.failed || status == DownloadTaskStatus.canceled) {
          isDownloading.value = false;
          _activeTaskId = null;
        } else if (status == DownloadTaskStatus.running) {
          isDownloading.value = true;
          _activeTaskId = id;
        }
      }
    });
  }

  Future<void> _checkRunningTasks() async {
    final tasks = await FlutterDownloader.loadTasks();
    if (tasks != null && tasks.isNotEmpty) {
      final runningTask = tasks.firstWhereOrNull(
              (t) => t.status == DownloadTaskStatus.running || t.status == DownloadTaskStatus.enqueued
      );
      if (runningTask != null) {
        _activeTaskId = runningTask.taskId;
        isDownloading.value = true;
        downloadProgress.value = (runningTask.progress > 0 ? runningTask.progress : 0) / 100;
        final model = LlamaService.instance.availableModels.firstWhereOrNull((m) => m.filename == runningTask.filename);
        if (model != null) currentDownloadingModel.value = model.name;
      }
    }
  }

  Future<void> downloadModel(ModelInfo model) async {
    if (isDownloading.value) return;
    if (Platform.isAndroid && (await Permission.notification.request()).isDenied) return;

    final directory = await _llamaService.getLocalModelsPath();
    final taskId = await FlutterDownloader.enqueue(
      url: model.url, savedDir: directory, fileName: model.filename, showNotification: true,
    );

    if (taskId != null) {
      _activeTaskId = taskId;
      isDownloading.value = true;
      currentDownloadingModel.value = model.name;
    }
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();

    nThreads.value = prefs.getInt('nThreads') ?? 4;
    nCtx.value = prefs.getInt('nCtx') ?? 0;
    nGpuLayers.value = prefs.getInt('nGpuLayers') ?? 0;
    useGpu.value = prefs.getBool('useGpu') ?? false;

    String savedPath = prefs.getString('selected_model_path') ?? "";
    String savedDraftPath = prefs.getString('selected_draft_model_path') ?? "";
    String savedVisionPath = prefs.getString('selected_vision_model_path') ?? ""; // --- NEW ---

    bool attemptedMainLoad = false;
    if (savedPath.isNotEmpty && File(savedPath).existsSync()) {
      attemptedMainLoad = true;
      await loadModel(savedPath);
    }

    bool attemptedDraftLoad = false;
    if (savedDraftPath.isNotEmpty && File(savedDraftPath).existsSync()) {
      attemptedDraftLoad = true;
      await loadDraftModel(savedDraftPath);
    }

    bool attemptedVisionLoad = false;
    if (savedVisionPath.isNotEmpty && File(savedVisionPath).existsSync()) {
      attemptedVisionLoad = true;
      await loadVisionModel(savedVisionPath);
    }

    await _refreshDownloadedModels(
        autoSelectMain: !attemptedMainLoad,
        autoSelectDraft: !attemptedDraftLoad,
        autoSelectVision: !attemptedVisionLoad
    );
  }

  Future<void> _refreshDownloadedModels({bool autoSelectMain = true, bool autoSelectDraft = true, bool autoSelectVision = true}) async {
    downloadedModels.value = await _llamaService.getDownloadedModels();

    if (autoSelectMain && selectedModelPath.value.isEmpty && mainModels.isNotEmpty) {
      await loadModel(mainModels.first.path);
    }

    if (autoSelectDraft && selectedDraftModelPath.value.isEmpty && draftModels.isNotEmpty) {
      await loadDraftModel(draftModels.first.path);
    }

    if (autoSelectVision && selectedVisionModelPath.value.isEmpty && visionModels.isNotEmpty) {
      await loadVisionModel(visionModels.first.path);
    }
  }

  void updateHardwareSettings(int threads, int contextSize, int gpuLayers, bool gpuEnabled) {
    nThreads.value = threads;
    nCtx.value = contextSize;
    nGpuLayers.value = gpuLayers;
    useGpu.value = gpuEnabled;
  }

  Future<void> applyHardwareSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('nThreads', nThreads.value);
    await prefs.setInt('nCtx', nCtx.value);
    await prefs.setInt('nGpuLayers', nGpuLayers.value);
    await prefs.setBool('useGpu', useGpu.value);

    // Reload all currently loaded models with new hardware settings
    if (selectedModelPath.value.isNotEmpty) {
      await loadModel(selectedModelPath.value);
    }
    if (selectedDraftModelPath.value.isNotEmpty) {
      await loadDraftModel(selectedDraftModelPath.value);
    }
    if (selectedVisionModelPath.value.isNotEmpty) {
      await loadVisionModel(selectedVisionModelPath.value);
    }
  }

  Future<void> loadModel(String path) async {
    try {
      Get.snackbar("Loading", "Initializing Main Engine...");
      await _llamaService.initModel(
          path,
          nCtx: nCtx.value > 0 ? nCtx.value : null,
          nThreads: nThreads.value,
          nGpuLayers: useGpu.value ? (nGpuLayers.value == 0 ? 10 : nGpuLayers.value) : 0
      );
      isModelLoaded.value = _llamaService.isInitialized.value;
      if (isModelLoaded.value) {
        selectedModelPath.value = path;
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('selected_model_path', path);
        Get.snackbar("Ready", "Main Model Initialized successfully.", backgroundColor: Colors.green.withOpacity(0.5));
      }
    } catch (e) {
      isModelLoaded.value = false;
      selectedModelPath.value = "";
      Get.snackbar("Initialization Failed", e.toString().replaceAll("Exception: ", ""),
          backgroundColor: Colors.redAccent,
          duration: const Duration(seconds: 5),
          snackPosition: SnackPosition.BOTTOM
      );
    }
  }

  Future<void> loadDraftModel(String path) async {
    try {
      Get.snackbar("Loading", "Initializing Draft Engine...");
      await _llamaService.initDraftModel(
          path,
          nCtx: nCtx.value > 0 ? nCtx.value : null,
          nThreads: nThreads.value,
          nGpuLayers: useGpu.value ? (nGpuLayers.value == 0 ? 10 : nGpuLayers.value) : 0
      );
      isDraftModelLoaded.value = _llamaService.isDraftInitialized.value;
      if (isDraftModelLoaded.value) {
        selectedDraftModelPath.value = path;
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('selected_draft_model_path', path);
        Get.snackbar("Ready", "Draft Model Initialized successfully.", backgroundColor: Colors.green.withOpacity(0.5));
      }
    } catch (e) {
      isDraftModelLoaded.value = false;
      selectedDraftModelPath.value = "";
      Get.snackbar("Initialization Failed", e.toString().replaceAll("Exception: ", ""),
          backgroundColor: Colors.orangeAccent,
          snackPosition: SnackPosition.BOTTOM
      );
    }
  }

  // --- NEW: Load Vision Model ---
  Future<void> loadVisionModel(String path) async {
    try {
      Get.snackbar("Loading", "Initializing Vision Projector...");
      await _llamaService.initVision(path);
      isVisionModelLoaded.value = _llamaService.isVisionInitialized.value;
      if (isVisionModelLoaded.value) {
        selectedVisionModelPath.value = path;
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('selected_vision_model_path', path);
        Get.snackbar("Ready", "Vision Model Initialized successfully.", backgroundColor: Colors.green.withOpacity(0.5));
      }
    } catch (e) {
      isVisionModelLoaded.value = false;
      selectedVisionModelPath.value = "";
      Get.snackbar("Initialization Failed", e.toString().replaceAll("Exception: ", ""),
          backgroundColor: Colors.purpleAccent,
          snackPosition: SnackPosition.BOTTOM
      );
    }
  }

  Future<void> deleteModel(File file) async {
    downloadedModels.removeWhere((f) => f.path == file.path);

    if (selectedModelPath.value == file.path) {
      _llamaService.dispose();
      isModelLoaded.value = false;
      selectedModelPath.value = "";
      (await SharedPreferences.getInstance()).remove('selected_model_path');
    }

    if (selectedDraftModelPath.value == file.path) {
      isDraftModelLoaded.value = false;
      selectedDraftModelPath.value = "";
      (await SharedPreferences.getInstance()).remove('selected_draft_model_path');
    }

    if (selectedVisionModelPath.value == file.path) {
      isVisionModelLoaded.value = false;
      selectedVisionModelPath.value = "";
      (await SharedPreferences.getInstance()).remove('selected_vision_model_path');
    }

    if (file.existsSync()) {
      await file.delete();
    }
    await _refreshDownloadedModels(autoSelectMain: false, autoSelectDraft: false, autoSelectVision: false);
  }
}