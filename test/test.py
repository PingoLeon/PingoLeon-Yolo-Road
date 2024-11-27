import cv2
from ultralytics import YOLO
import os
import logging
from pathlib import Path
import shutil

#! Settings
model_version_name = "yolo11n_grayscale_test"  # testv11_grayscale or testv11_color
#!######################################################

# Configuration des logs
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def process_images(model_version_name):
    # Charger le modèle avec les meilleurs poids
    model = YOLO(f'runs/detect/{model_version_name}/weights/best.pt')
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path('test/test/dataset_test_output')
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parcourir les images dans le dossier d'entrée
    input_dir = Path('test/test/dataset_test')
    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Lire l'image
            img = cv2.imread(str(img_path))
            
            #si le modèle ne contient pas grayscale dans son nom, ne pas convrtir les images en gris avant de les traiter
            if "grayscale" in model_version_name:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Faire la prédiction
            results = model.predict(source=img, conf=0.40)
            
            # Sauvegarder l'image annotée
            output_path = output_dir / img_path.name
            results[0].save(str(output_path))
            
            logging.info(f"Traité : {img_path.name}")

if __name__ == '__main__':
    process_images(model_version_name)