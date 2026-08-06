package com.example.app

fun greet(name: String): String = "Hello, $name!"

fun main(args: Array<String>) {
    val name = args.firstOrNull() ?: "world"
    println(greet(name))
}
