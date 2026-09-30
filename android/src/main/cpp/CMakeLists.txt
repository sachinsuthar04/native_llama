cmake_minimum_required(VERSION 3.18.1)
project("native_llama")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization and defines
add_compile_options(-O3 -fexceptions -frtti)

# --- Configuration for local llama.cpp ---
set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
set(GGML_VULKAN_PERF_FA OFF CACHE BOOL "Disable Flash Attention" FORCE)

# Disable unnecessary heavy builds from the local repo
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server" FORCE)

# We link against llama.cpp's `common` library (common_chat_* / chat templates),
# but it defaults to LLAMA_STANDALONE (OFF here, since llama.cpp is added via
# add_subdirectory). Force it on so libcommon is built and -lcommon resolves.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common utils library" FORCE)

# `common` optionally links libcurl (for downloading models from a URL), which
# isn't available in the Android NDK sysroot. We download models ourselves, so
# disable it to avoid an unresolved libcurl dependency.
set(LLAMA_CURL OFF CACHE BOOL "Disable libcurl model download" FORCE)

# --- Force compiler to ignore hardware audio frameworks ---
add_definitions(-DMA_NO_OPENSL=1)
add_definitions(-DMA_NO_AAUDIO=1)
add_definitions(-DMA_NO_AVFOUNDATION=1)
add_definitions(-DMA_NO_COREAUDIO=1)

include(FetchContent)

# Vulkan and SPIR-V headers, required to build ggml-vulkan for Android.
#
# PINNED DELIBERATELY. These were previously `GIT_TAG main`, which means an
# upstream commit to a Khronos repository can break — or silently change — this
# build with no local change and nothing in the diff to explain it. That is a
# bad trade for headers: they move constantly and we gain nothing from tracking
# them live.
#
# Both pins are the exact revisions the first successful Android build of
# ai_rpg_game resolved and linked against (2026-08-07, arm64-v8a, Q1_0 Vulkan).
# Vulkan-Headers landed exactly on a release tag; SPIRV-Headers sat four commits
# past vulkan-sdk-1.4.357.0 and so has no tag of its own — hence the raw SHA,
# which is in any case the more honest way to say "this exact tree".
#
# To bump: move both together, rebuild, and confirm libggml-vulkan.so still
# links. Do not bump one alone.
FetchContent_Declare(VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.4.359)
FetchContent_MakeAvailable(VulkanHeaders)

FetchContent_Declare(SPIRVHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG 942fe4b988359a0750b79f0ae7ed735994d3147d)  # vulkan-sdk-1.4.357.0-4
FetchContent_MakeAvailable(SPIRVHeaders)

# 3. CRITICAL FIX: Point directly to the single source of truth in the iOS folder!
# Path traversal: cpp -> main -> src -> android -> plugin_root -> ios/shared_cpp
set(SHARED_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../ios/shared_cpp")

# Add the local directory to the build
# (Requires a secondary binary_dir argument since the source is outside the current CMake tree)
add_subdirectory("${SHARED_CPP_DIR}" llama_cpp_build)

# Apply Vulkan headers to the ggml-vulkan target
target_include_directories(ggml-vulkan PRIVATE
    "${vulkanheaders_SOURCE_DIR}/include"
    "${spirvheaders_SOURCE_DIR}/include"
)

# Grab all your custom MTMD model files
file(GLOB MTMD_MODEL_SRCS "${SHARED_CPP_DIR}/tools/mtmd/models/*.cpp")

# Add our custom bridge files
add_library(native_llama SHARED
    native-lib.cpp
    build-info.cpp
    "${SHARED_CPP_DIR}/tools/mtmd/clip.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd-helper.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd-image.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/mtmd-audio.cpp"
    "${SHARED_CPP_DIR}/tools/mtmd/deprecation-warning.cpp"
    ${MTMD_MODEL_SRCS}
)

# Explicitly specify include directories mapping to the unified local folder
target_include_directories(native_llama PRIVATE
    .
    "${SHARED_CPP_DIR}/include"
    "${SHARED_CPP_DIR}/common"
    "${SHARED_CPP_DIR}/src"
    "${SHARED_CPP_DIR}/ggml/include"
    "${SHARED_CPP_DIR}/ggml/src"
    "${SHARED_CPP_DIR}/tools/mtmd"
    "${SHARED_CPP_DIR}/vendor"
)

find_library(log-lib log)
# Link against the locally built llama library. `llama-common` provides the
# common_chat_* (Jinja chat-template) symbols used by startGeneration. NOTE:
# newer llama.cpp renamed the old `common` target to `llama-common`.
target_link_libraries(native_llama llama llama-common dl vulkan ${log-lib})