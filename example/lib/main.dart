import 'dart:isolate';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:flutter_downloader/flutter_downloader.dart';

import 'core/theme/app_theme.dart';
import 'core/bindings/initial_binding.dart';
import 'features/home/view/home_view.dart';

@pragma('vm:entry-point')
void downloadCallback(String id, int status, int progress) {
  final SendPort? send = IsolateNameServer.lookupPortByName('downloader_send_port');
  send?.send([id, status, progress]);
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await FlutterDownloader.initialize(
      debug: true,
      ignoreSsl: true
  );

  FlutterDownloader.registerCallback(downloadCallback);

  runApp(const NativeLlamaExampleApp());
}

class NativeLlamaExampleApp extends StatelessWidget {
  const NativeLlamaExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Native Llama',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      initialBinding: InitialBinding(),
      home: const HomeView(),
      defaultTransition: Transition.cupertino,
    );
  }
}