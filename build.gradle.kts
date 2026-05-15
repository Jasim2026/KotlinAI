plugins {
    id("com.android.application") version "8.4.0" apply false
    
    // Upgraded from 2.0.0 to 2.2.21 to support LiteRT's modern dependencies
    id("org.jetbrains.kotlin.android") version "2.2.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.21" apply false
}