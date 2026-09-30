import 'package:get/get.dart';
import '../../features/home/controller/home_controller.dart';
import '../../features/assistant/controller/assistant_controller.dart';
import '../../features/settings/controller/settings_controller.dart';

class InitialBinding extends Bindings {
  @override
  void dependencies() {
    Get.lazyPut(() => HomeController());
    Get.lazyPut(() => AssistantController());
    Get.lazyPut(() => SettingsController());
  }
}