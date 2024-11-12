import os
import cv2
from ultralytics import YOLO

def verify_annotations_and_results():
    # Charger le modèle de détection YOLOv8
    model = YOLO('C:/Users/leon/Downloads/computer-vision/runs/detect/yolov11_Detection5/weights/best.pt')

    # Dossier contenant les images de test
    img_folder = 'C:/Users/leon/Downloads/computer-vision/dataset/speed_signs/train/images'
    output_folder = 'C:/Users/leon/Downloads/computer-vision/output_images'
    os.makedirs(output_folder, exist_ok=True)

    # Parcourir toutes les images du dossier
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        
        # Ignorer les dossiers
        if os.path.isdir(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur: Impossible de charger l'image depuis {img_path}")
            continue

        # Effectuer la détection
        results = model(img)

        # Dessiner des rectangles autour des objets détectés
        detected = False
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convertir le tenseur en liste et accéder aux coordonnées
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detected = True

        # Sauvegarder l'image uniquement si des objets ont été détectés
        if detected:
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, img)
            print(f"Image avec objets détectés sauvegardée à {output_path}")
        else:
            print(f"Aucun objet détecté dans l'image {img_name}")

if __name__ == '__main__':
    verify_annotations_and_results()