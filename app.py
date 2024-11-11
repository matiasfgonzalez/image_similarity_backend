import os
import cv2
import numpy as np
from scipy.spatial import distance

def calculate_histogram(image_path):
    """Calcula el histograma de color de una imagen en escala de grises."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def calculate_similarity(hist1, hist2):
    """Calcula la similitud entre dos histogramas utilizando la distancia coseno."""
    return 1 - distance.cosine(hist1, hist2)

def find_most_similar_image(test_image_path, folder_path):
    test_hist = calculate_histogram(test_image_path)
    if test_hist is None:
        print(f"Error al cargar la imagen de prueba en {test_image_path}")
        return

    similarities = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        stored_hist = calculate_histogram(image_path)
        if stored_hist is not None:
            similarity = calculate_similarity(test_hist, stored_hist)
            similarities.append((filename, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

if __name__ == "__main__":
    test_image_path = "images/test.jpg"
    folder_path = "images/stored_images"
    similarities = find_most_similar_image(test_image_path, folder_path)
    
    if similarities:
        print("Imágenes similares (ordenadas por similitud):")
        for filename, similarity in similarities:
            print(f"{filename}: {similarity:.4f}")
    else:
        print("No se encontraron imágenes similares o no se pudieron cargar.")
