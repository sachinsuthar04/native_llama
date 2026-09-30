group = "com.timebox.native_llama"
version = "1.0-SNAPSHOT"

buildscript {
    val kotlinVersion = "2.2.20"
    repositories {
        google()
        mavenCentral()
    }

    dependencies {
        classpath("com.android.tools.build:gradle:8.11.1")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.library")
    id("kotlin-android")
}

android {
    namespace = "com.timebox.native_llama"

    compileSdk = 36
    // 1. Matched NDK version from your app
    ndkVersion = "28.2.13676358"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin")
        }
        getByName("test") {
            java.srcDirs("src/test/kotlin")
        }
    }

    defaultConfig {
        minSdk = 28

        // 2. Transferred ABI filters to keep the plugin AAR size small
        ndk {
            abiFilters.add("arm64-v8a")
            // Uncomment the line below if you also need to run this on desktop/x86 Android Emulators
            // abiFilters.add("x86_64")
        }

        externalNativeBuild {
            cmake {
                cppFlags(
                    "-std=c++17",
                    "-DGGML_USE_VULKAN=1",
                    "-DGGML_VULKAN_PERF_FA=0"
                )

                arguments(
                    "-DGGML_USE_VULKAN=1",
                    "-DGGML_VULKAN_PERF_FA=0"
                )
            }
        }
    }

    // 3. Linked the CMakeLists.txt to compile llama.cpp
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // 4. Added legacy packaging to ensure .so files aren't compressed (crucial for JNI extraction)
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    testOptions {
        unitTests {
            isIncludeAndroidResources = true
            all {
                it.useJUnitPlatform()

                it.outputs.upToDateWhen { false }

                it.testLogging {
                    events("passed", "skipped", "failed", "standardOut", "standardError")
                    showStandardStreams = true
                }
            }
        }
    }
}

dependencies {
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("org.mockito:mockito-core:5.0.0")
}