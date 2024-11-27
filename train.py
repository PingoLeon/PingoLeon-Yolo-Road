from ultralytics import YOLO
import torch
import logging

# Configuration des logs
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

#! Settings 
model_name = "yolo11s" # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x (after s version, training time increases significantly)
dataset = "grayscale" # grayscale or color
version = "test" # test, production, or whatever part of the process you're at
epoch_number = 50 # number of epochs, the higher the better but the longer it takes
batch_size = 8 # influence on the speed of the training and vram usage
#!######################################################

if dataset == "color":
    data = 'dataset/french-road-signs-v11-color/data.yaml'
else:
    data = 'dataset/french-road-signs-v11-grayscale/data.yaml'

# Charger le modèle de détection YOLOv8
model = YOLO(f'dataset/{model_name}.pt')

# Vérifier CUDA
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Utilisation de : {device}")
if torch.cuda.is_available():
    logging.info(f"GPU détectée : {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA version : {torch.version.cuda}")

# Lancer l'entraînement
if __name__ == '__main__':
    results = model.train(
        data=f'{data}',
        epochs=epoch_number,
        imgsz=640,
        batch=batch_size,
        name=f'{model_name}_{dataset}_{version}_',
        cache=True,
        device=device,
        exist_ok=True  # permettre l'écrasement des fichiers existants
    )