from ultralytics import YOLO

# Charger le modèle de détection YOLOv8
model = YOLO('yolo11n.pt')

# Lancer l'entraînement
if __name__ == '__main__':
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='yolov11_Detection',
        cache=True,
    )