import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:file_picker/file_picker.dart'; // --- MODIFIED: Changed from image_picker ---
import '../../../core/services/llama_service.dart';
import '../../../core/services/database_service.dart';
import '../../../core/services/semantic_cache_service.dart';
import '../../../core/models/chat_model.dart';

class AssistantController extends GetxController {
  final LlamaService _llamaService = LlamaService.instance;
  final DatabaseService _dbService = DatabaseService.instance;
  final SemanticCacheService _cacheService = SemanticCacheService();

  var messages = <ChatMessage>[].obs;
  var currentSessionId = Rxn<int>();
  var isTyping = false.obs;
  var isThinkingEnabled = true.obs;
  final ScrollController scrollController = ScrollController();

  // --- MODIFIED: Renamed to attachedMediaPaths ---
  var attachedMediaPaths = <String>[].obs;
  // -----------------------------------

  var temperature = 0.7.obs;
  var topK = 40.obs;
  var topP = 0.9.obs;

  var isVisionInitialized = false.obs;

  @override
  void onInit() {
    super.onInit();
    // Bind to the service's reactive variable
    isVisionInitialized.bindStream(_llamaService.isVisionInitialized.stream);
    isVisionInitialized.value = _llamaService.isVisionInitialized.value;
    createNewChat();
  }

  Future<void> createNewChat() async {
    currentSessionId.value = await _dbService.createChatSession("Plugin Test Chat");
    messages.clear();
    attachedMediaPaths.clear();
  }

  void scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (scrollController.hasClients) {
        scrollController.animateTo(
            scrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut);
      }
    });
  }

  // --- MODIFIED: Media Picker Method (Supports Images & Audio) ---
  Future<void> pickMedia() async {
    FilePickerResult? result = await FilePicker.pickFiles(
      allowMultiple: true,
      type: FileType.custom,
      // Restrict to formats that MTMD (stb_image & miniaudio) natively supports
      allowedExtensions: ['png', 'jpg', 'jpeg', 'bmp', 'wav', 'mp3', 'm4a', 'flac'],
    );

    if (result != null) {
      // Filter out null paths just to be safe
      attachedMediaPaths.addAll(result.paths.whereType<String>());
      scrollToBottom();
    }
  }

  void removeAttachedMedia(int index) {
    attachedMediaPaths.removeAt(index);
  }
  // --------------------------------

  void sendMessage(String text) async {
    // Allow sending just media without text
    if (text.isEmpty && attachedMediaPaths.isEmpty) return;
    int sessionId = currentSessionId.value!;

    // Create a local copy of paths so we can clear the UI immediately
    final List<String> currentMessageMedia = List.from(attachedMediaPaths);

    messages.add(ChatMessage(
      sessionId: sessionId,
      role: "user",
      content: text,
      createdAt: DateTime.now(),
      mediaPaths: currentMessageMedia,
    ));
    await _dbService.saveChatMessage(sessionId, "user", text, mediaPaths: currentMessageMedia);

    // Clear UI state
    attachedMediaPaths.clear();
    isTyping.value = true;
    scrollToBottom();

    // Check Initialization FIRST
    if (!_llamaService.isInitialized.value) {
      messages.add(ChatMessage(
          sessionId: sessionId,
          role: "assistant",
          content: "Model not initialized. Please load a model in Settings.",
          createdAt: DateTime.now()
      ));
      isTyping.value = false;
      scrollToBottom();
      return;
    }

    // Check if media is attached but Vision/Audio projector is NOT initialized
    if (currentMessageMedia.isNotEmpty && !_llamaService.isVisionInitialized.value) {
      messages.add(ChatMessage(
          sessionId: sessionId,
          role: "assistant",
          content: "I cannot process media because a Projector (MMPROJ) was not loaded in Settings.",
          createdAt: DateTime.now()
      ));
      isTyping.value = false;
      scrollToBottom();
      return;
    }

    // Disable Semantic Cache when using media for safety (embeddings are usually text-only)
    if (currentMessageMedia.isEmpty) {
      final cachedResponse = await _cacheService.getCachedResponse(text);
      if (cachedResponse != null) {
        messages.add(ChatMessage(sessionId: sessionId, role: "assistant", content: "[CACHED] $cachedResponse", createdAt: DateTime.now()));
        isTyping.value = false;
        scrollToBottom();
        return;
      }
    }

    final assistantIndex = messages.length;
    messages.add(ChatMessage(
      sessionId: sessionId,
      role: "assistant",
      content: "",
      createdAt: DateTime.now(),
      expectTags: isThinkingEnabled.value,
    ));

    String systemPrompt;
    if (isThinkingEnabled.value) {
      systemPrompt =
          "You are a helpful assistant. Provide a brief, high-level chain of thought inside <thought> tags before the final answer. Keep the thinking process concise and avoid unnecessary length. If the reasoning is complex, break it down into clear, successive steps. Then, provide the final answer inside <final_answer> tags. Example: <thought>1. Analysis... 2. Steps...</thought><final_answer>The answer.</final_answer>";
    } else {
      systemPrompt = "You are a helpful assistant. Provide a direct and concise answer without any internal reasoning tags.";
    }

    List<Map<String, String>> chatHistory = [
      {"role": "system", "text": systemPrompt}
    ];

    // --- MODIFIED: Inject <|media_pad|> tokens so the model knows where to "look" or "listen" ---
    String finalContent = text;
    if (currentMessageMedia.isNotEmpty) {
      String mediaTags = currentMessageMedia.map((_) => "<|media_pad|>").join("\n");
      finalContent = "$mediaTags\n$text";
    }

    for (var m in messages.sublist(0, assistantIndex)) {
      chatHistory.add({"role": m.role, "text": m.content});
    }

    // Replace the last message text with the media-padded version
    chatHistory.removeLast();
    chatHistory.add({"role": "user", "text": finalContent});

    String fullResponse = "";
    try {
      await for (final token in _llamaService.generateResponse(
        chatHistory,
        mediaPaths: currentMessageMedia, // --- MODIFIED: Pass mediaPaths to Native ---
        temperature: temperature.value,
        topK: topK.value,
        topP: topP.value,
      )) {
        fullResponse += token;
        messages[assistantIndex].updateContent(fullResponse);
        messages.refresh();
        scrollToBottom();
      }

      messages[assistantIndex].updateContent(fullResponse, isFinalized: true);
      messages.refresh();
      await _dbService.saveChatMessage(sessionId, "assistant", fullResponse);

      if (currentMessageMedia.isEmpty) {
        await _cacheService.cacheResponse(text, fullResponse);
      }

    } catch (e) {
      messages[assistantIndex].updateContent("Error: $e");
    } finally {
      isTyping.value = false;
    }
  }

  void stopGeneration() {
    _llamaService.abortGeneration();
    isTyping.value = false;
  }
}