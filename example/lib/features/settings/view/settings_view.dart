import 'dart:io';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controller/settings_controller.dart';
import '../../../core/services/llama_service.dart';

class SettingsView extends GetView<SettingsController> {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plugin Settings'),
        actions: [
          IconButton(icon: const Icon(Icons.file_download), onPressed: controller.importModelFile),
        ],
      ),
      body: Obx(() => ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildSectionHeader('Active Configuration'),
          Card(
            color: Colors.blueGrey[900],
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  _buildModelDropdown(
                    label: "Main LLM Engine", icon: Icons.psychology,
                    value: controller.selectedModelPath.value,
                    options: controller.mainModels,
                    onChanged: (p) => controller.loadModel(p!),
                  ),
                  const Divider(height: 32, color: Colors.white10),
                  _buildModelDropdown(
                    label: "Draft Model (Speculative Decoding)", icon: Icons.bolt,
                    value: controller.selectedDraftModelPath.value,
                    options: controller.draftModels,
                    onChanged: (p) => controller.loadDraftModel(p!),
                  ),
                  const Divider(height: 32, color: Colors.white10),
                  _buildModelDropdown(
                    label: "Vision Projector (MMPROJ)", icon: Icons.visibility,
                    value: controller.selectedVisionModelPath.value,
                    options: controller.visionModels,
                    onChanged: (p) => controller.loadVisionModel(p!),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          _buildSectionHeader('Hardware Overrides'),
          Card(
            color: Colors.grey[900],
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text("GPU Inference (Vulkan)", style: TextStyle(fontWeight: FontWeight.bold)),
                      Switch(
                        value: controller.useGpu.value,
                        onChanged: (val) => controller.updateHardwareSettings(
                            controller.nThreads.value,
                            controller.nCtx.value,
                            controller.nGpuLayers.value,
                            val
                        ),
                        activeColor: Colors.blueAccent,
                      ),
                    ],
                  ),
                  if (controller.useGpu.value) ...[
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text("GPU Layers", style: TextStyle(fontSize: 14)),
                        Text(controller.nGpuLayers.value == -1 ? "Max (Full Offload)" : "${controller.nGpuLayers.value}",
                            style: const TextStyle(color: Colors.blueAccent)),
                      ],
                    ),
                    Slider(
                      value: controller.nGpuLayers.value == -1 ? 99 : controller.nGpuLayers.value.toDouble(),
                      min: 0,
                      max: 99,
                      divisions: 99,
                      activeColor: Colors.blueAccent,
                      onChanged: (val) {
                        int layers = val.toInt();
                        if (layers >= 99) layers = -1;
                        controller.updateHardwareSettings(
                            controller.nThreads.value,
                            controller.nCtx.value,
                            layers,
                            controller.useGpu.value
                        );
                      },
                    ),
                    const Text("Caution: High values may crash mobile GPUs.",
                        style: TextStyle(fontSize: 11, color: Colors.orangeAccent)),
                    const SizedBox(height: 16),
                  ],
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text("CPU Threads", style: TextStyle(fontWeight: FontWeight.bold)),
                      Text("${controller.nThreads.value}", style: const TextStyle(color: Colors.blueAccent)),
                    ],
                  ),
                  Slider(
                    value: controller.nThreads.value.toDouble(),
                    min: 1,
                    max: 16,
                    divisions: 15,
                    activeColor: Colors.blueAccent,
                    onChanged: (val) => controller.updateHardwareSettings(
                        val.toInt(),
                        controller.nCtx.value,
                        controller.nGpuLayers.value,
                        controller.useGpu.value
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text("Context Size", style: TextStyle(fontWeight: FontWeight.bold)),
                      Text(controller.nCtx.value == 0 ? "Auto (RAM Based)" : "${controller.nCtx.value} tokens", style: const TextStyle(color: Colors.blueAccent)),
                    ],
                  ),
                  Slider(
                    value: controller.nCtx.value.toDouble(),
                    min: 0,
                    max: 16384,
                    divisions: 8,
                    activeColor: Colors.blueAccent,
                    onChanged: (val) {
                      int snappedVal = 0;
                      if (val > 0 && val <= 3072) snappedVal = 2048;
                      else if (val > 3072 && val <= 6144) snappedVal = 4096;
                      else if (val > 6144 && val <= 12288) snappedVal = 8192;
                      else if (val > 12288) snappedVal = 16384;

                      if (snappedVal != controller.nCtx.value) {
                        controller.updateHardwareSettings(
                            controller.nThreads.value,
                            snappedVal,
                            controller.nGpuLayers.value,
                            controller.useGpu.value
                        );
                      }
                    },
                  ),
                  const Padding(
                    padding: EdgeInsets.only(top: 8.0),
                    child: Text("Set to 0 to let the engine auto-calculate limits to prevent crashes.", style: TextStyle(fontSize: 12, color: Colors.white54)),
                  ),
                  const SizedBox(height: 24),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: controller.applyHardwareSettings,
                      icon: const Icon(Icons.sync),
                      label: const Text("Apply & Reload Models"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blueAccent,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: StadiumBorder(side: BorderSide.none),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),
          _buildSectionHeader('Available to Download'),
          ...LlamaService.instance.availableModels.map((m) => _buildModelDownloadCard(m)),
          const SizedBox(height: 24),
          _buildSectionHeader('Local Storage'),
          ...controller.downloadedModels.map((file) => _buildLocalModelCard(file)),
          if (controller.isDownloading.value) ...[
            const SizedBox(height: 24),
            _buildSectionHeader('Downloading: ${controller.currentDownloadingModel.value}'),
            LinearProgressIndicator(value: controller.downloadProgress.value),
          ],
        ],
      )),
    );
  }

  Widget _buildModelDropdown({required String label, required IconData icon, required String value, required List<File> options, required void Function(String?) onChanged}) {
    final currentValue = options.any((f) => f.path == value) ? value : null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(children: [Icon(icon, size: 16, color: Colors.blueAccent), const SizedBox(width: 8), Text(label)]),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          value: currentValue,
          dropdownColor: Colors.blueGrey[900],
          isExpanded: true,
          decoration: InputDecoration(
              filled: true,
              fillColor: Colors.black26,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
              hintText: "Select a model..."
          ),
          items: options.map((f) => DropdownMenuItem(value: f.path, child: Text(f.path.split('/').last, overflow: TextOverflow.ellipsis))).toList(),
          onChanged: options.isEmpty ? null : onChanged,
        ),
      ],
    );
  }

  Widget _buildModelDownloadCard(ModelInfo model) {
    final isDownloaded = controller.downloadedModels.any((f) => f.path.endsWith(model.filename));
    return Card(
      color: Colors.grey[900],
      child: ListTile(
        title: Text(model.name, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text("${model.sizeGB}GB"),
        trailing: isDownloaded ? const Icon(Icons.check_circle, color: Colors.green) : IconButton(icon: const Icon(Icons.download, color: Colors.blue), onPressed: () => controller.downloadModel(model)),
      ),
    );
  }

  Widget _buildLocalModelCard(File file) {
    return Card(
      color: Colors.grey[900],
      child: ListTile(
        title: Text(file.path.split('/').last),
        trailing: IconButton(icon: const Icon(Icons.delete, color: Colors.red), onPressed: () => controller.deleteModel(file)),
      ),
    );
  }

  Widget _buildSectionHeader(String title) => Padding(padding: const EdgeInsets.only(bottom: 8), child: Text(title.toUpperCase(), style: const TextStyle(color: Colors.blueAccent, fontWeight: FontWeight.bold)));
}