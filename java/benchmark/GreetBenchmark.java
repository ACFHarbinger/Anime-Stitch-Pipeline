package com.example.app.benchmark;

import com.example.app.Main;

/**
 * Simple micro-benchmark, run manually, e.g.:
 * {@code javac -d /tmp/out $(find ../src/main/java -name '*.java') GreetBenchmark.java && \
 *  java -cp /tmp/out:. com.example.app.benchmark.GreetBenchmark}
 */
public final class GreetBenchmark {

    private GreetBenchmark() {
    }

    public static void main(String[] args) {
        int iterations = 1_000_000;
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            Main.greet("world");
        }
        long elapsed = System.nanoTime() - start;
        System.out.printf("greet(): %d ns/op%n", elapsed / iterations);
    }
}
