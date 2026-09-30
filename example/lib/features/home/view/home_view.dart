import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controller/home_controller.dart';
import '../../assistant/view/assistant_view.dart';
import '../../settings/view/settings_view.dart';

class HomeView extends GetView<HomeController> {
  const HomeView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Obx(() => IndexedStack(
        index: controller.selectedIndex.value,
        children: const [AssistantView(), SettingsView()],
      )),
      bottomNavigationBar: Obx(() => BottomNavigationBar(
        currentIndex: controller.selectedIndex.value,
        onTap: controller.changeTab,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.chat), label: 'Test Chat'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: 'Plugin Settings'),
        ],
      )),
    );
  }
}