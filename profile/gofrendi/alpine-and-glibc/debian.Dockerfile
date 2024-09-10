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
