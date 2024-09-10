docker build -f debian.Dockerfile -t my-debian-go-app .
docker build -f alpine.Dockerfile -t my-alpine-go-app .
echo "RUN DEBIAN"
docker run my-debian-go-app
echo "RUN ALPINE"
docker run my-alpine-go-app