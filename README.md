# CREAR IMAGEN

docker build -t image-similarity-service .

# CREAR CONTENEDOR

docker run -d -p 8000:8000 -v "$(pwd)/images:/app/images" image-similarity-service
