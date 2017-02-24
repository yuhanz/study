package main

import "net"
import "fmt"
import "bufio"

func handleMessage(messageChan <-chan string) {
  for {
    message := <- messageChan
    fmt.Print("Message Received:", message)
  }
}

func tcpServerLoop(port int, messageChan chan<- string) {

  portStr := fmt.Sprintf(":%d", port)
  fmt.Println("Launching server at port " + portStr + " ...")

  // listen on all interfaces
  ln, _ := net.Listen("tcp", portStr)

  // accept connection on port
  conn, _ := ln.Accept()
  reader := bufio.NewReader(conn)

  // run loop forever (or until ctrl-c)
  for {
    // will listen for message to process ending in newline (\n)
    message, _ := reader.ReadString('\n')
    // output message received
    messageChan <- string(message)

    // send new string back to client
    conn.Write([]byte(message + "\n"))

    if message == "" {
        conn.Close()
        conn, _ = ln.Accept()
        reader = bufio.NewReader(conn)
    }
  }
}

func main() {
  messageQueue := make(chan string, 10)
  go handleMessage(messageQueue)
  tcpServerLoop(8081, messageQueue)
}
