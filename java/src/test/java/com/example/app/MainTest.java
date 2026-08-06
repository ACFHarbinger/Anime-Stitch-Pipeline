package com.example.app;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MainTest {

    @Test
    void greetReturnsExpectedMessage() {
        assertEquals("Hello, Dev-Repo-Template!", Main.greet("Dev-Repo-Template"));
    }

    @Test
    void greetHandlesDefaultCase() {
        assertEquals("Hello, world!", Main.greet("world"));
    }
}
