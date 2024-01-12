# python-object-detection-with-yolo-and-opencv

## Table of Contents

+ [Summary](#summary)
+ [References](#references)
+ [How to use](#how-to-use)

## Summary

Simple Object detector app utilizing trained YOLO v3 or YOLO v4 CNN models.

## References 

- [darknet Github repository](https://github.com/pjreddie/darknet)
- [yolov4_darknet Github repository](https://github.com/kiyoshiiriemon/yolov4_darknet)
- [YOLO: Real Time Object detection](https://pjreddie.com/darknet/yolo/) - The tutorial also explains how supervised learning can be applied for the YOLO architecture to generate a custom trained model with `.weights` (weights of the nodes determined by backward pass), `.cfg` (contains network architecture details, hyperparameters, input and output configuration, training settings) and `.names` (model classes utilized for training) files.

## How to use

### Perequisite

- Ensure you have a WebCam

### Install the pip packages

```sh
pip install -r requirements.txt
```

### Retrieve files required for object detection

Following links can be utilized to download the `.weights` files:

- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

Copy those weights in the `cfg` folder in this project.

### (Optional) Copy `.cfg` and `.names` files

Following steps can be considered in order to retrieve the`.cfg` and `.names` files:

- Clone both mentioned Github repositories in the Reference section
- In `darknet/cfg` and `yolov4_darknet/cfg` you can find the cfg files which you can copy to the cfg folder if not already existing 
- In `darknet/data` and `yolov4_darknet/data` you can find the `.names` (list of classes) files, e.g. `coco.names` 

### Run the app

Execute the object detection application utilizing any of the downloaded YOLO weights:

```sh
python object_detector_app.py --help
# Object detection with yolov4
python object_detector_app.py --model yolov4-tiny
```
