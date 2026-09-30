import 'dart:io';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import '../controller/assistant_controller.dart';
import '../../../core/models/chat_model.dart';

class AssistantView extends GetView<AssistantController> {
  const AssistantView({super.key});

  @override
  Widget build(BuildContext context) {
    final TextEditingController textController = TextEditingController();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Plugin Test Chat'),
        actions: [
          Obx(() => Row(
            children: [
              const Text("Thinking", style: TextStyle(fontSize: 12)),
              Switch(
                value: controller.isThinkingEnabled.value,
                onChanged: (val) => controller.isThinkingEnabled.value = val,
                activeColor: Colors.blueAccent,
              ),
            ],
          )),
          IconButton(icon: const Icon(Icons.refresh), onPressed: controller.createNewChat)
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Obx(() => ListView.builder(
              controller: controller.scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: controller.messages.length,
              itemBuilder: (context, index) {
                final msg = controller.messages[index];
                return msg.role == 'user' ? _buildUserMsg(context, msg) : _buildAssistantMsg(context, msg);
              },
            )),
          ),
          Obx(() => controller.isTyping.value ? const LinearProgressIndicator(minHeight: 2) : const SizedBox.shrink()),

          // --- MODIFIED: Media Attachment Preview Bar ---
          Obx(() {
            if (controller.attachedMediaPaths.isEmpty) return const SizedBox.shrink();
            return Container(
              height: 80,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: controller.attachedMediaPaths.length,
                itemBuilder: (context, index) {
                  final path = controller.attachedMediaPaths[index];
                  final isAudio = path.toLowerCase().endsWith('.wav') ||
                      path.toLowerCase().endsWith('.mp3') ||
                      path.toLowerCase().endsWith('.m4a') ||
                      path.toLowerCase().endsWith('.flac');

                  return Stack(
                    children: [
                      Container(
                        margin: const EdgeInsets.only(right: 8),
                        width: 60,
                        decoration: BoxDecoration(
                          color: Colors.grey[800],
                          borderRadius: BorderRadius.circular(8),
                          // Only load FileImage if it's an actual image
                          image: isAudio ? null : DecorationImage(
                            image: FileImage(File(path)),
                            fit: BoxFit.cover,
                          ),
                        ),
                        // Show audio icon if it's an audio file
                        child: isAudio
                            ? const Center(child: Icon(Icons.audiotrack, color: Colors.blueAccent, size: 32))
                            : null,
                      ),
                      Positioned(
                        right: 4,
                        top: 0,
                        child: GestureDetector(
                          onTap: () => controller.removeAttachedMedia(index),
                          child: Container(
                            decoration: const BoxDecoration(shape: BoxShape.circle, color: Colors.black54),
                            child: const Icon(Icons.close, size: 16, color: Colors.white),
                          ),
                        ),
                      )
                    ],
                  );
                },
              ),
            );
          }),
          // -----------------------------------------

          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              children: [
                // --- MODIFIED: Attachment Button (Hidden if no vision projector) ---
                Obx(() => controller.isVisionInitialized.value
                    ? IconButton(
                        onPressed: controller.isTyping.value ? null : controller.pickMedia,
                        icon: const Icon(Icons.attach_file, color: Colors.grey),
                      )
                    : const SizedBox.shrink()),
                Expanded(
                  child: TextField(
                    controller: textController,
                    decoration: InputDecoration(
                      hintText: 'Test the LLM...',
                      filled: true, fillColor: Colors.grey[900],
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(30), borderSide: BorderSide.none),
                    ),
                  ),
                ),
                IconButton(
                  onPressed: () {
                    if (controller.isTyping.value) {
                      controller.stopGeneration();
                    } else {
                      controller.sendMessage(textController.text);
                      textController.clear();
                    }
                  },
                  icon: Obx(() => Icon(controller.isTyping.value ? Icons.stop_circle : Icons.send, color: Colors.blueAccent)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUserMsg(BuildContext context, ChatMessage msg) {
    return Align(
      alignment: Alignment.centerRight,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (msg.mediaPaths != null && msg.mediaPaths!.isNotEmpty)
            _buildMediaGrid(msg.mediaPaths!),
          Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: Colors.blueAccent, borderRadius: BorderRadius.circular(16)),
            child: Text(msg.answerText, style: const TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _buildMediaGrid(List<String> paths) {
    return Container(
      constraints: const BoxConstraints(maxWidth: 250),
      margin: const EdgeInsets.only(bottom: 8),
      child: Wrap(
        spacing: 4,
        runSpacing: 4,
        alignment: WrapAlignment.end,
        children: paths.map((path) {
          final isAudio = path.toLowerCase().endsWith('.wav') ||
              path.toLowerCase().endsWith('.mp3') ||
              path.toLowerCase().endsWith('.m4a') ||
              path.toLowerCase().endsWith('.flac');

          return Container(
            width: paths.length == 1 ? 200 : 80,
            height: paths.length == 1 ? 150 : 80,
            decoration: BoxDecoration(
              color: Colors.grey[800],
              borderRadius: BorderRadius.circular(8),
              image: isAudio ? null : DecorationImage(
                image: FileImage(File(path)),
                fit: BoxFit.cover,
              ),
            ),
            child: isAudio
                ? const Center(child: Icon(Icons.audiotrack, color: Colors.blueAccent, size: 32))
                : null,
          );
        }).toList(),
      ),
    );
  }

  Widget _buildAssistantMsg(BuildContext context, ChatMessage msg) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (msg.thoughtText.isNotEmpty || (msg.role == 'assistant' && msg.expectTags && msg.answerText.isEmpty))
            _ThinkingBlock(thoughtText: msg.thoughtText),
          if (msg.answerText.isNotEmpty)
            Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: Colors.grey[800], borderRadius: BorderRadius.circular(16)),
              child: Markdown(data: msg.answerText, shrinkWrap: true, physics: const NeverScrollableScrollPhysics(), padding: EdgeInsets.zero),
            ),
        ],
      ),
    );
  }
}

class _ThinkingBlock extends StatefulWidget {
  final String thoughtText;
  const _ThinkingBlock({required this.thoughtText});

  @override
  State<_ThinkingBlock> createState() => _ThinkingBlockState();
}

class _ThinkingBlockState extends State<_ThinkingBlock> {
  bool _isExpanded = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white10),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          InkWell(
            onTap: () => setState(() => _isExpanded = !_isExpanded),
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.psychology_outlined, size: 20, color: Colors.blueAccent),
                  const SizedBox(width: 8),
                  const Text("Thinking Process", style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.blueAccent)),
                  const SizedBox(width: 8),
                  if (widget.thoughtText.isEmpty)
                    const SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.blueAccent),
                    )
                  else
                    Icon(_isExpanded ? Icons.expand_less : Icons.expand_more, size: 20, color: Colors.grey),
                ],
              ),
            ),
          ),
          if (_isExpanded)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
              child: Markdown(
                data: widget.thoughtText,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                padding: EdgeInsets.zero,
              ),
            ),
        ],
      ),
    );
  }
}