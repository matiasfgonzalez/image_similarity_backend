import os
import cv2
import numpy as np
from scipy.spatial import distance
from fastapi import FastAPI, File, UploadFile
from typing import List, Dict, Union

app = FastAPI()

def calculate_histogram(image_data):
    """Calcula el histograma de una imagen en escala de grises."""
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def calculate_similarity(hist1, hist2):
    """Calcula la similitud entre dos histogramas usando la distancia coseno."""
    return 1 - distance.cosine(hist1, hist2)

def find_most_similar_image(uploaded_image_data, folder_path) -> List[Dict[str, Union[str, float]]]:
    """Encuentra las imágenes más similares en la carpeta dada y devuelve una lista de diccionarios con nombres y similitudes."""
    uploaded_hist = calculate_histogram(uploaded_image_data)
    if uploaded_hist is None:
        return []

    similarities = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        with open(image_path, "rb") as f:
            stored_image_data = f.read()
            stored_hist = calculate_histogram(stored_image_data)
            if stored_hist is not None:
                similarity = calculate_similarity(uploaded_hist, stored_hist)
                # Convertimos numpy.float32 a float para evitar problemas de serialización
                similarities.append({"filename": filename, "similarity": float(similarity)})

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities

@app.post("/compare-image")
async def compare_image(file: UploadFile = File(...)):
    image_data = await file.read()
    folder_path = "images/stored_images"
    similarities = find_most_similar_image(image_data, folder_path)

    if similarities:
        return {"similar_images": similarities}
    else:
        return {"message": "No similar images found or could not process the uploaded image."}
