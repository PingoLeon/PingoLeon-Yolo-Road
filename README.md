# YOLOroad

This project aims to

## Run & Test

#### Structure of the project :

* train.py
  * to launch a training set based on parameters
* dataset folder
  * Where 2 versions of the dataset is present, including image and labels
  * one dataset is the grayscaled version of the first one, to consider performance differences
* test folder
  * Used to run scripts after the training
    * video.py to launch the webcam and hold signs in front to try any fine-tuned model
    * test.py to test the fine-tuned model on a whole folder (dataset_test) of images

#### Prerequisites

Start by cloning the project on your machine

```
git clone https://github.com/PingoLeon/Yolo-Road.git
```

You will need a recent version of Python (like [3.12](https://www.python.org/downloads/release/python-3127/https://www.python.org/downloads/release/python-3127/ "Download Python")) with multiple dependencies :

* Ultralytics
* Torch
* CUDA toolkit (If using Nvidia GPU)

  * try `nvcc --version` in a terminal to ensure that CUDA toolkit is installed
  * also try running `torch.cuda.is_available()` to ensure it is available, and thus ensure maximum possible speed on your Nvidia GPU

  ```python
  >>> import torch
  >>> torch.cuda.is_available()
  True
  ```

Versions used for this project :

* Ultralytics 8.3.13
* Torch 2.4.1+cu118 (checkout [this page](https://pytorch.org/get-started/locally/ "Pytorch download page") to get the install command corresponding your system requirements)
* CUDA 11.8 (download [here](https://developer.nvidia.com/cuda-11-8-0-download-archive "Nvidia website"))

```
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> [!WARNING]
> You will need to run the scripts from the original repo folder reference, else the paths will be messed up

Once you're done with all the requirements, just hit `python train.py` to start the training. Training parameters are in the beginning of the `train.py` file.

For testing, place your test images in `test/test_images/dataset_test` folder to be able to run the `test/test.py` code

## Performance metrics

The original dataset was made by scraping online images from google images or other sources. The images may be protected by copyright. The labeling process was made on [Roboflow](https://universe.roboflow.com/image-understanding/panneaux-v2 "Dataset Labeled"), and dispatched on a basis : 97% train set / 3% valid set / 1% test. The valid test could be higher but since I didn't have many images I wanted to emphasize the accuracy of the training part.

YOLO11 models for detection comes in multiple models :

| Model   | Size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
| ------- | ------------- | ------------ | -------------------- | ------------------------ | ---------- | --------- |
| YOLO11n | 640           | 39.5         | 56.1 ± 0.8          | 1.5 ± 0.0               | 2.6        | 6.5       |
| YOLO11s | 640           | 47.0         | 90.0 ± 1.2          | 2.5 ± 0.0               | 9.4        | 21.5      |
| YOLO11m | 640           | 51.5         | 183.2 ± 2.0         | 4.7 ± 0.1               | 20.1       | 68.0      |
| YOLO11l | 640           | 53.4         | 238.6 ± 1.4         | 6.2 ± 0.1               | 25.3       | 86.9      |
| YOLO11x | 640           | 54.7         | 462.8 ± 6.7         | 11.3 ± 0.2              | 56.9       | 194.9     |

I ran multiple tests on a few lightweight models. Most of the tests were on nano, small and medium models, at 50 or 100 epochs per runs. For reference, the training ran on a RTX 3050Ti Laptop GPU, and the longest training took 6 hours on the medium model on 100 epochs. Going with models like large or x would have too much parameters for my GPU to handle, despite having a greater accuracy.

I first ran tests on the nano model, which is the most lightweight but has only 2.6M parameters, and thus making a lot of errors after the training.

I then went on the small model, which provided greater accuracy while still making a good amount of errors in the recognition.

I tried to run the medium model on 100 epochs, but it took a huge amount of time. However, having 20.1M parameters hugely improbed the accuracy.

I choosed to make the dataset in 2 versions, one original and one grayscaled version. My impression is that the detection on a a grayscale basis has a greater accuracy on the color one since the model doesn't have to deal with 3 RGB channels informations, hence improving the detection of forms, like specific logos on signs or numbers.

Here is some performance comparisons based on the runs :

YOLO11 small model, 100 epochs - grayscale dataset

<figure style="text-align: left;">
  <p style="font-family: arial; margin: 0;">Confusion matrix normalized</p>
  <img src="runs/detect/runs/detect/yolo11s_grayscale_test_/results.png" alt="global metrics" width="400"/>
</figure>

<figure style="text-align: left;">
  <p style="font-family: arial; margin: 0;">Predictions on validation data</p>
  <img src="runs/detect/detect/yolo11s_grayscale_test_/val_batch1_pred.jpg" alt="predictions on validation data" width="400"/>
</figure>

YOLO11 small model, 100 epochs - color dataset

<figure style="text-align: left;">
  <p style="font-family: arial; margin: 0;">Confusion matrix normalized</p>
  <img src="runs/detect/runs/detect/yolo11s_color_test_/results.png" alt="global metrics" width="400"/>
</figure>

<figure style="text-align: left;">
  <p style="font-family: arial; margin: 0;">Predictions on validation data</p>
  <img src="runs/detect/detect/yolo11s_color_test_/val_batch1_pred.jpg" alt="predictions on validation data" width="400"/>
</figure>

# References

[](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/README.md#references)

Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11 (11.0.0) [Computer software]. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).

Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., Duerig, T., & Ferrari, V. (2020). The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale. IJCV.

# Tasks

tweaks : dataset labeled and formed thanks to roboflow, labelIMG being too hard to understand. Compare Ultralytics YOLO differences betwwen latest v11 version and v? version, to se the differences and improvements in performance the model has made.

output : lightweight vision model (.pt, pytorch format) to run python scripts with it, such as giving it a video or anything else and be able to detect most of the signs visible (in a reasonable quality for the image / video)

At first, I tried to use LabelIMG to label the images, but I ended up using Roboflow since it had the option to have the dataset suited for YOLOV11.

The first run I made was with YOLOV11n, which is the smallest model with only 2.5M parameters. The curves at first seemed promising, but when testing the model on new pictures, the detection was really deceiving.

I then tried the YOLOV11s model, which is the second smallest model with 8.7M parameters. The curves were better, and so were the detections. The model was able to detect the signs, but it was still not perfect.

Unfortunately, I made the choice to run the model on my own computer, with a 3050 Ti Laptop GPU, which is quite slow at a small rate of 50 epochs.

| Modèle | Taille (pixels) | mAPval 50-95 | Vitesse CPU ONNX (ms) | Vitesse T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
| ------- | --------------- | ------------ | --------------------- | -------------------------- | ---------- | --------- |
| YOLO11n | 640             | 39.5         | 56.1 ± 0.8           | 1.5 ± 0.0                 | 2.6        | 6.5       |
| YOLO11s | 640             | 47.0         | 90.0 ± 1.2           | 2.5 ± 0.0                 | 9.4        | 21.5      |
| YOLO11m | 640             | 51.5         | 183.2 ± 2.0          | 4.7 ± 0.1                 | 20.1       | 68.0      |
| YOLO11l | 640             | 53.4         | 238.6 ± 1.4          | 6.2 ± 0.1                 | 25.3       | 86.9      |
| YOLO11x | 640             | 54.7         | 462.8 ± 6.7          | 11.3 ± 0.2                | 56.9       | 194.9     |
