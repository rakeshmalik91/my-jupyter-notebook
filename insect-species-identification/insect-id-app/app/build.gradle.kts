plugins {
    alias(libs.plugins.android.application)
//    id("com.google.gms.google-services")
}

android {
    namespace = "com.rakeshmalik.insectid"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.rakeshmalik.insectid"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
//    packaging {
//        resources {
//            excludes += "META-INF/INDEX.LIST"
//            excludes += "META-INF/DEPENDENCIES"
//            excludes += "META-INF/io.netty.versions.properties"
//        }
//    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    implementation(libs.pytorch.android)
    implementation(libs.pytorch.android.torchvision)
    implementation(libs.gson)
    implementation(libs.ucrop)
    implementation(libs.okhttp)
//    implementation(platform(libs.firebase.bom))
//    implementation(libs.firebase.analytics)
//    implementation(libs.firebase.admin)
//    implementation(libs.google.cloud.storage)
//    implementation(libs.google.services)
}
