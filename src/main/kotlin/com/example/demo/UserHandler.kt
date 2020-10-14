package com.example.demo

import org.springframework.http.HttpStatus
import org.springframework.http.MediaType.APPLICATION_JSON
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.reactive.function.server.ServerRequest
import org.springframework.web.reactive.function.server.ServerResponse
import reactor.core.publisher.Mono

@RestController
class UserHandler(val userRepo: UserRepo) {

    @RequestMapping("/")
    fun test(): String = "hello"

    @RequestMapping("/2")
    fun test2(): Mono<String> = Mono.just("hello2")


    fun allUsers(req: ServerRequest): Mono<ServerResponse> = ServerResponse
            .ok()
            .body(userRepo.allUsers(), User::class.java)

    data class Error        (val code: String, val msg: String)
    data class ErrorResponse(val error: Error)

}