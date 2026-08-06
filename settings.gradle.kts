/*
 * Root Gradle settings — ties together the Kotlin module. The actual
 * project lives in kotlin/ (see kotlin/build.gradle.kts).
 */

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
    }
}

rootProject.name = "dev-repo-template"

include(":kotlin")
