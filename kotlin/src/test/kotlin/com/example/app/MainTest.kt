package com.example.app

import kotlin.test.Test
import kotlin.test.assertEquals

class MainTest {
    @Test
    fun `greet returns expected message`() {
        assertEquals("Hello, Dev-Repo-Template!", greet("Dev-Repo-Template"))
    }

    @Test
    fun `greet handles default case`() {
        assertEquals("Hello, world!", greet("world"))
    }
}
