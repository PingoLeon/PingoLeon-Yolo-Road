from ultralytics import YOLO
import cv2
import time

def process_webcam():
    # Charger le modèle entraîné
    model = YOLO('runs/detect/yolov11_Detection5/weights/best.pt')
    
    # Initialiser la webcam (0 = webcam par défaut)
    cap = cv2.VideoCapture(0)
    
    # Vérifier si la webcam est ouverte correctement
    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la webcam")
        return
        
    print("Appuyez sur 'q' pour quitter")
    
    while True:
        # Lire une frame de la webcam
        success, frame = cap.read()
        if not success:
            break
            
        # Faire la détection
        results = model(frame)
        
        # Dessiner les boîtes de détection
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Récupérer la confiance et la classe
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Filtrer les détections avec confiance > 0.6
                    if conf > 0.6:
                        # Obtenir les coordonnées
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Dessiner le rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Afficher l'ID de classe et la confiance
                        label = f"Class {cls} ({conf:.2f})"
                        cv2.putText(frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Afficher la frame
        cv2.imshow('Detection en temps reel', frame)
        
        # Quitter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_webcam()