package com.example.demo

import org.springframework.stereotype.Component
import reactor.core.publisher.Flux

interface UserRepo {
    fun allUsers(): Flux<User>
}

@Component
class UserRepoImpl(): UserRepo {

    internal fun toUser() = User(
            user_id = 1,
            name    = "myName",
            email   = "myEmail")



    override fun allUsers(): Flux<User> = Flux.just(toUser())
}