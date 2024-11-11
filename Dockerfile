# Usar la imagen oficial de Python
FROM python:3.10-slim

# Crear el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requirements y instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Ejecutar el servidor de FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
