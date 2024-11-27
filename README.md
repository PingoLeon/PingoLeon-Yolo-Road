# YOLOroad

This project aims to 

## Getting Started

Clone the project on your machine

```
git clone https://github.com/Alfred0404/Smart_Fridge_Project_Code.git
```

### Prerequisites

[](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/README.md#prerequisites)

You need [Python &gt;3.11](https://www.python.org/downloads/) and some dependencies:

* [Flask](https://flask.palletsprojects.com/en/3.0.x/)
* [Ollama](https://github.com/ollama/ollama-python)
* [OpenCV](https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/index.html#)
* [Ultralytics](https://docs.ultralytics.com/quickstart/#install-ultralytics)
* [EasyOCR](https://pypi.org/project/easyocr/)

```
pip install opencv-python ollama Flask ultralytics easyocr
```

## Training

[](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/README.md#training)

The model was trained using a [custom dataset](https://app.roboflow.com/fridgeinventorydetection/fridge_inventory_detection/1). Images come from our personnal fridge, and Google Image. Every image has been labeled by hand.

Here's some stats about the model so far:

Confusion matrix normalized

[![predictions on validation data](https://github.com/Alfred0404/Smart_Fridge_Project_Code/raw/computer_vision/runs/detect/train/confusion_matrix_normalized.png)](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/runs/detect/train/confusion_matrix_normalized.png)

Global metrics

[![predictions on validation data](https://github.com/Alfred0404/Smart_Fridge_Project_Code/raw/computer_vision/runs/detect/train/results.png)](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/runs/detect/train/results.png)

Predictions on validation data

[![predictions on validation data](https://github.com/Alfred0404/Smart_Fridge_Project_Code/raw/computer_vision/runs/detect/train/val_batch1_pred.jpg)](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/runs/detect/train/val_batch1_pred.jpg)

It's only a first training test, which is very conclusive and reinforces the idea of continuing along this path. There is still a lot to do.

### Cuda

[](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/README.md#cuda)

Cuda has accelerated the learning process, enabling tensorflow to use the Nvidia GPU to compute the learning data.

* Install dependencies by generating your command [here](https://pytorch.org/get-started/locally/), you should get something like that: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
* Check if cuda is properly downloaded
  ```python
  >>> import torch
  >>> torch.cuda.is_available()
  True
  ```
* If you're struggling, this [stackoverflow discussion](https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud) helped get it to work.

## Authors

[](https://github.com/Alfred0404/Smart_Fridge_Project_Code/blob/computer_vision/README.md#authors)

* **Alfred de Vulpian** - [Alfred0404](https://github.com/Alfred0404)
* **Clément d'Alberto** - [Clement-dl](https://github.com/https://github.com/Clement-dl)
* **Alara Tanguy** - [AlTang01](https://github.com/AlTang01)

See the list of [contributors](https://github.com/Alfred0404/Smart_Fridge_Project_Code/contributors) who participated in this project.

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
