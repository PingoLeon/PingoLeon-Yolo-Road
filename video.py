import cv2
from ultralytics import YOLO
import torch
import logging

# Configuration des logs
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Charger le modèle entraîné
model = YOLO('runs/detect/testv114/weights/best.pt')  # utilise le meilleur poids entraîné

# Configuration de la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

# Vérifier si la webcam est ouverte correctement
if not cap.isOpened():
    logging.error("Erreur: Impossible d'ouvrir la webcam")
    exit()

while True:
    # Lire une frame de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Reconvertir en BGR pour la compatibilité avec YOLO
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Faire la prédiction
    results = model.predict(source=gray_frame_bgr, conf=0.40)
    
    # Visualiser les résultats
    annotated_frame = results[0].plot()
    
    # Afficher le résultat
    cv2.imshow('Détection de panneaux routiers', annotated_frame)
    
    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()