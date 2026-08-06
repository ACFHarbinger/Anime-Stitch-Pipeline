/*
 * Root build file — shared configuration for all Gradle subprojects (currently
 * just `kotlin/`; add `java/` here too if it's ever migrated from Maven).
 */

plugins {
    kotlin("jvm") version "2.0.20" apply false
    id("org.jlleitschuh.gradle.ktlint") version "12.1.1" apply false
}

allprojects {
    group = "com.example.dev-repo-template"
    version = "0.1.0"

    repositories {
        mavenCentral()
    }
}
