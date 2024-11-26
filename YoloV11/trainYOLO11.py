from ultralytics import YOLO

# Charger le modèle de détection YOLOv8
model = YOLO('YoloV11/yolo11n.pt')

# Lancer l'entraînement
if __name__ == '__main__':
    results = model.train(
        data='YoloV11/data11.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='testv11',
        cache=True,
    )