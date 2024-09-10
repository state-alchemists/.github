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