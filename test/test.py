import shutil
from pathlib import Path
import logging
import cv2
import yaml
from ultralytics import YOLO

# Configuration des logs
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return data['names']

def process_images(model_version_name):
    # Charger le modèle avec les meilleurs poids
    model = YOLO(f'runs/detect/{model_version_name}/weights/best.pt')
    
    # Charger les noms des classes à partir du fichier YAML
    if "grayscale" in model_version_name:
        yaml_path = 'dataset/french-road-signs-v11-grayscale/data.yaml'  # Chemin vers le fichier YAML
    else:
        yaml_path = 'dataset/french-road-signs-v11-color/data.yaml'
    class_names = load_class_names(yaml_path)
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = Path('test/test_images/dataset_test_output')
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parcourir les images dans le dossier d'entrée
    input_dir = Path('test/test_images/dataset_test')
    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Lire l'image en couleur
            img_color = cv2.imread(str(img_path))
            
            # Vérifier si le modèle contient "grayscale" dans son nom
            if "grayscale" in model_version_name:
                # Convertir l'image en niveaux de gris pour le traitement
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                
                # Faire la prédiction sur l'image en niveaux de gris
                results = model.predict(source=img_gray, conf=0.25)
            else:
                # Faire la prédiction sur l'image en couleur
                results = model.predict(source=img_color, conf=0.25)
            
            # Dessiner les cases et les labels sur l'image en couleur originale
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label_index = int(box.cls[0])  # Assurez-vous que le label est correctement récupéré
                    confidence = box.conf[0]  # Assurez-vous que la confiance est correctement récupérée
                    label = class_names[label_index]
                    text = f'{label}: {confidence:.2f}'
                    cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_color, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Sauvegarder l'image annotée
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), img_color)
            
            logging.info(f"Traité : {img_path.name}")

if __name__ == '__main__':
    model_version_name = 'yolo11s_color_test_'  # Exemple de nom de version
    process_images(model_version_name)