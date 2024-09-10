# Alpine and GlibC

Alpine is commonly used as base image because of its small size. However, Alpine comes with a few compatibility problem:

- It uses `musl` instead of `glibc`.
- It has `libc6-compat` for `glibc` compatibility.

Thus, any executable that relies on `glibc` and not supported by `libc6-compat` will surely failed.

# The code

Here is a golang code that run a C program relying on `glibc`

```go
package main

/*
#include <malloc.h>
#include <stdio.h>

void test_mallopt() {
	printf("Attempting to set M_MMAP_THRESHOLD in glibc...\n");
    // M_MMAP_THRESHOLD is glibc-specific and may not be supported by musl
    if (mallopt(M_MMAP_THRESHOLD, 64 * 1024) == 0) {
        printf("Failed to set M_MMAP_THRESHOLD (glibc-specific)\n");
    } else {
        printf("Successfully set M_MMAP_THRESHOLD\n");
    }
	printf("Okey");
}
*/
import "C"

import "fmt"

func main() {
	fmt.Println("Testing mallopt function...")
	C.test_mallopt() // This should fail on Alpine Linux
	fmt.Println("Done")
}

```

# The Debian Dockerfile

For the first experiment, let's try to wrap the application with Debian based container:

```dockerfile
# Stage 1: Build the Go application using a full Golang image
FROM golang:bookworm

# Set the Current Working Directory inside the container
WORKDIR /app

# Copy the Go modules manifest and cache dependencies
COPY go.mod ./
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Enable CGO and build the Go application
RUN CGO_ENABLED=1 go build -o main .

# Command to run the executable
CMD ["./main"]
```

You can run the container with no problem.

```bash
docker build -f debian.Dockerfile -t my-debian-go-app .
docker run my-debian-go-app
```

# The Alpine Dockerfile

As for alpine, we will try to use a multistage dockerfile:

```dockerfile
# Stage 1: Build the Go application using a full Golang image
FROM golang:bookworm AS builder

# Set the Current Working Directory inside the container
WORKDIR /app

# Copy the Go modules manifest and cache dependencies
COPY go.mod ./
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Enable CGO and build the Go application
RUN CGO_ENABLED=1 go build -o main .

# Stage 2: Create a lightweight image to run the Go application
FROM alpine:latest

# Set the Current Working Directory inside the container
WORKDIR /app

# Install necessary runtime dependencies (if needed for CGO)
RUN apk add --no-cache libc6-compat

# Copy the executable from the builder stage
COPY --from=builder /app/main .

# Command to run the executable
CMD ["./main"]
```

You can run the application as follow:

```bash
docker build -f alpine.Dockerfile -t my-alpine-go-app .
docker run my-alpine-go-app
```

This unfortunately yield error:

```
Error relocating /app/main: mallopt: symbol not found
```

Apparently `musl` with `libc6-compat` cannot recognize `mallopt`

# Conclusion

Using alpine as docker image is not always a good decision. Despite of it's small size, it has a few compatibility issue.

In general, I prefer to use `debian:slim` as my base image. But if you choose to use `alpine`, please be aware of this compatibility issue. It rarely happened, but the possibility is still there.