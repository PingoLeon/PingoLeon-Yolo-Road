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
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 opencv-python pyyaml
```

Additionals requirements : 

> [!WARNING]
> You will need to run the scripts from the original repo folder reference, else the paths will be messed up

Once you're done with all the requirements, just hit `python train.py` to start the training. Training parameters are in the beginning of the `train.py` file.

For testing, place your test images in `test/test_images/dataset_test` folder to be able to run the `test/test.py` code. you have to put the name of the run version yourself to choose which model you would like to test
