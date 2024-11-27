from ultralytics import YOLO
import cv2
import time
import yaml

def load_class_names(yaml_file):
    """Charge les noms des classes depuis le fichier YAML"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']

def process_webcam():
    # Charger le modèle entraîné
    model = YOLO('runs/detect/testv11/weights/best.pt')
    
    # Charger les noms des classes
    class_names = load_class_names('./YoloV11/data11.yaml')  # Ajuster le chemin selon votre configuration
    
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la webcam")
        return
        
    print("Appuyez sur 'q' pour quitter")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        results = model(frame)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Récupérer le nom de la classe
                    class_name = class_names[cls_id]
                    
                    # Obtenir les coordonnées de la boîte
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Dessiner la boîte
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Afficher le nom de la classe et la confiance
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Detection en temps reel', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_webcam()