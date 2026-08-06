package com.example.app.benchmark

import com.example.app.greet
import kotlin.system.measureNanoTime

/** Simple micro-benchmark, run manually via `./gradlew run -PmainClass=...`. */
fun main() {
    val iterations = 1_000_000
    val elapsed = measureNanoTime {
        repeat(iterations) { greet("world") }
    }
    println("greet(): ${elapsed / iterations} ns/op")
}
