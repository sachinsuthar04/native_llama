Pod::Spec.new do |s|
  s.name             = 'native_llama'
  s.version          = '1.0.1'
  s.summary          = 'On-device LLM plugin'
  s.description      = 'A highly optimized, hardware-accelerated Flutter plugin.'
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Timebox' => 'explorearitra@gmail.com' }
  s.source           = { :path => '.' }

  # 1. Using a physical copy for absolute reliability with CocoaPods
  s.source_files = [
    'Classes/**/*.{h,m,mm,swift,cpp,c}',
    'shared_cpp/include/*.h',
    'shared_cpp/ggml/include/*.h',
    'shared_cpp/src/*.{h,cpp,c}',
    'shared_cpp/src/models/*.{h,cpp}',
    'shared_cpp/common/*.{h,cpp,c}',
    'shared_cpp/common/jinja/*.{h,cpp}',
    'shared_cpp/ggml/src/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-cpu/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-cpu/arch/arm/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-cpu/llamafile/*.{h,cpp}',
    'shared_cpp/vendor/**/*.{h,c,cpp}',
    'shared_cpp/ggml/src/ggml-metal/*.{h,m,mm,cpp}',
    'shared_cpp/tools/mtmd/**/*.{h,cpp,c}'
  ]

  # Exclude ALL standalone CLI and Debug tools to prevent duplicate main() symbols.
  s.exclude_files = [
    'shared_cpp/tools/mtmd/mtmd-cli.cpp',
    'shared_cpp/tools/mtmd/debug/mtmd-debug.cpp'
  ]

  s.public_header_files = 'Classes/**/*.h'

  s.project_header_files = [
    'shared_cpp/**/*.h',
    'shared_cpp/**/*.hpp'
  ]

  s.dependency 'Flutter'
  s.platform = :ios, '17.0'

  # Metal shader: native_sd also vendors ggml and bundles its own (merged)
  # ggml-metal.metal at the app-bundle root. To coexist, we (1) inline the
  # #included headers so this shader compiles at runtime, and (2) bundle it under
  # a UNIQUE name (nl-ggml-metal.metal). Our ggml-metal loader is patched to look
  # for that name, and its class is symbol-prefixed (NL_GGMLMetalClass), so the
  # two metal backends load their own shaders with no collision.
  s.prepare_command = <<-CMD
    cd shared_cpp/ggml/src/ggml-metal
    awk '/#include "ggml-common.h"/{system("cat ../ggml-common.h");next}1' ggml-metal.metal > nl-tmp.metal
    awk '/#include "ggml-metal-impl.h"/{system("cat ggml-metal-impl.h");next}1' nl-tmp.metal > nl-ggml-metal.metal
    rm nl-tmp.metal
  CMD

  s.resources = ['shared_cpp/ggml/src/ggml-metal/nl-ggml-metal.metal']

  s.compiler_flags = '-fno-objc-arc -DMA_NO_AVFOUNDATION=1 -DMA_NO_COREAUDIO=1'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',

    'MTL_PREPROCESSOR_DEFINITIONS' => 'GGML_METAL_HAS_BF16=1',
    'MTL_LANGUAGE_REVISION' => 'Metal31',

    'OTHER_LDFLAGS' => '$(inherited) -framework Metal -framework Foundation',

    # --- CRITICAL FIX: Undefine the broken Apple cache line macro and Force Obj-C++ ---
    # The -include namespaces this plugin's vendored ggml/gguf/stb symbols (prefix
    # nl_) so they don't collide at the app link with native_sd's ggml / MediaPipe's
    # stb (~448 duplicate symbols otherwise). See shared_cpp/ggml_symbol_prefix.h.
    'OTHER_CPLUSPLUSFLAGS' => '$(inherited) -fno-modules -x objective-c++ -U__cpp_lib_hardware_interference_size -include "$(PODS_TARGET_SRCROOT)/shared_cpp/ggml_symbol_prefix.h"',
    'OTHER_CFLAGS' => '$(inherited) -include "$(PODS_TARGET_SRCROOT)/shared_cpp/ggml_symbol_prefix.h"',

    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',

    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/common"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src/models"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/arch/arm"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/llamafile"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/vendor"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/tools/mtmd"',
      '"$(PODS_ROOT)/Headers/Public/native_llama"'
    ].join(' '),

    'USER_HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/common"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/src/models"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/arch/arm"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-cpu/llamafile"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-metal"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/vendor"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/tools/mtmd"'
    ].join(' '),

    'MTL_HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/include"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src"',
      '"$(PODS_TARGET_SRCROOT)/shared_cpp/ggml/src/ggml-metal"'
    ].join(' '),

    'GCC_PREPROCESSOR_DEFINITIONS' => [
      '$(inherited)',
      'GGML_USE_METAL=1',
      'GGML_USE_ACCELERATE=1',
      'GGML_USE_CPU=1',
      'GGML_METAL_NDEBUG=1',
      'GGML_METAL_HAS_BF16=1',
      'GGML_VERSION="\\"4412\\""',
      'GGML_COMMIT="\\"82f7e77\\""'
    ].join(' ')
  }

  s.frameworks = 'Accelerate', 'Metal', 'MetalKit', 'MetalPerformanceShaders'
  s.swift_version = '5.0'
end