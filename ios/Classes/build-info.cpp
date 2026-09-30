#include "build-info.h"

int llama_build_number() {
    return 0;
}

const char * llama_commit() {
    return "unknown";
}

const char * llama_compiler() {
    return "clang";
}

const char * llama_build_target() {
    return "ios";
}

const char * llama_build_info() {
    return "manual-ios-build";
}

const char * LICENSES[] = { nullptr };