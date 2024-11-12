import os
import cv2
import numpy as np
from scipy.spatial import distance
from fastapi import FastAPI, File, UploadFile
from typing import List, Dict, Union
from fastapi.middleware.cors import CORSMiddleware
import cx_Oracle

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las solicitudes desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Configuración de la conexión a la base de datos Oracle
def get_oracle_connection():
    dsn = cx_Oracle.makedsn("hostname", "port", service_name="service_name")  # Reemplaza con tus datos
    return cx_Oracle.connect(user="your_username", password="your_password", dsn=dsn)

def calculate_histogram(image_data, color_space=cv2.COLOR_BGR2GRAY):
    """Calcula el histograma de una imagen en un espacio de color especificado."""
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    # Convertir la imagen al espacio de color deseado
    image = cv2.cvtColor(image, color_space)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def calculate_orb_features(image_data):
    """Calcula los descriptores ORB para la imagen proporcionada."""
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def calculate_similarity(hist1, hist2):
    """Calcula la similitud entre dos histogramas usando la distancia coseno."""
    return 1 - distance.cosine(hist1, hist2)

def compare_orb_features(desc1, desc2):
    """Compara descriptores ORB entre dos imágenes."""
    if desc1 is None or desc2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return len(matches) / max(len(desc1), len(desc2)) if desc1 is not None else 0.0

def get_images_from_db() -> List[Dict[str, Union[str, bytes]]]:
    """Obtiene las imágenes de la base de datos Oracle en formato BLOB."""
    connection = get_oracle_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT image_id, image_data FROM tbl_imagenes")
    images = []
    for image_id, image_data in cursor:
        images.append({
            "image_id": image_id,
            "image_data": image_data
        })

    cursor.close()
    connection.close()
    return images

def find_most_similar_image(uploaded_image_data) -> List[Dict[str, Union[str, float]]]:
    """Encuentra las imágenes más similares en la base de datos y devuelve una lista de diccionarios con nombres y similitudes."""
    # Calcular histogramas en escala de grises y color para la imagen cargada
    uploaded_hist_gray = calculate_histogram(uploaded_image_data, color_space=cv2.COLOR_BGR2GRAY)
    uploaded_hist_hsv = calculate_histogram(uploaded_image_data, color_space=cv2.COLOR_BGR2HSV)
    uploaded_orb_desc = calculate_orb_features(uploaded_image_data)

    similarities = []
    
    # Obtener imágenes de la base de datos
    images_from_db = get_images_from_db()

    for image in images_from_db:
        stored_image_data = image["image_data"]

        # Calcular histogramas y descriptores ORB para la imagen almacenada
        stored_hist_gray = calculate_histogram(stored_image_data, color_space=cv2.COLOR_BGR2GRAY)
        stored_hist_hsv = calculate_histogram(stored_image_data, color_space=cv2.COLOR_BGR2HSV)
        stored_orb_desc = calculate_orb_features(stored_image_data)
        
        # Calcular similitudes
        gray_similarity = calculate_similarity(uploaded_hist_gray, stored_hist_gray)
        hsv_similarity = calculate_similarity(uploaded_hist_hsv, stored_hist_hsv)
        orb_similarity = compare_orb_features(uploaded_orb_desc, stored_orb_desc)
        
        # Calcular un puntaje promedio
        average_similarity = (gray_similarity + hsv_similarity + orb_similarity) / 3
        similarities.append({
            "image_id": image["image_id"],
            "gray_similarity": float(gray_similarity),
            "hsv_similarity": float(hsv_similarity),
            "orb_similarity": float(orb_similarity),
            "average_similarity": float(average_similarity)
        })

    similarities.sort(key=lambda x: x["average_similarity"], reverse=True)
    return similarities

@app.post("/compare-image")
async def compare_image(file: UploadFile = File(...)):
    image_data = await file.read()

    similarities = find_most_similar_image(image_data)

    if similarities:
        return {"similar_images": similarities}
    else:
        return {"message": "No similar images found or could not process the uploaded image."}
