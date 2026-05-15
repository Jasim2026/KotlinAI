plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.jas.ai"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.jas.ai"
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }
    kotlinOptions {
        jvmTarget = "21"
    }
    
    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            // CRITICAL: Resolves conflicts between LiteRT-LM and TFLite Task Library
            pickFirsts += "**/libtensorflowlite_jni.so"
            pickFirsts += "**/libtensorflowlite_framework_jni.so"
        }
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += "**/libLiteRt.so"
            keepDebugSymbols += "**/libLiteRtClGlAccelerator.so"
            keepDebugSymbols += "**/liblitertlm_jni.so"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.2")
    implementation("androidx.activity:activity-compose:1.9.0")
    
    implementation(platform("androidx.compose:compose-bom:2024.05.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    
    // LiteRT-LM for gemma-1b-it-int4 (Generative)
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.11.0") 
    
    // MediaPipe Tasks Text for embedding (Replaces deprecated tensorflow-lite-task-text)
    implementation("com.google.mediapipe:tasks-text:0.10.14")
}