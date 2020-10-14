package com.example.demo

import org.springframework.context.annotation.Bean
import org.springframework.http.MediaType.APPLICATION_JSON
import org.springframework.stereotype.Component
import org.springframework.web.reactive.function.server.router

@Component
class Router {

    @Bean
    fun route(userHandler: UserHandler) = router {
        contentType(APPLICATION_JSON).nest {
            GET("/api/users/all", userHandler::allUsers)
        }
    }
}