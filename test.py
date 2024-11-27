import cv2
from ultralytics import YOLO
import os
import logging
from pathlib import Path

# Configuration des logs
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def process_images():
    # Charger le modèle avec les meilleurs poids
    model = YOLO('runs/detect/testv114/weights/best.pt')
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path('TestModel/dataset_test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parcourir les images dans le dossier d'entrée
    input_dir = Path('TestModel/dataset')
    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Lire l'image
            img = cv2.imread(str(img_path))
            
            # Convertir en niveaux de gris
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            
            # Faire la prédiction
            results = model.predict(source=gray_img_bgr, conf=0.40)
            
            # Sauvegarder l'image annotée
            output_path = output_dir / img_path.name
            results[0].save(str(output_path))
            
            logging.info(f"Traité : {img_path.name}")

if __name__ == '__main__':
    process_images()