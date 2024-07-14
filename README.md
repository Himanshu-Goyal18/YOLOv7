# YOLOv7 - Real-Time Object Detection
## 1. Overview
YOLOv7 (You Only Look Once, version 7) is the latest iteration in the YOLO family of object detection models, designed for high performance and real-time object detection. It offers state-of-the-art accuracy and speed, making it suitable for various applications like autonomous driving, surveillance, and more.

### 1.1 Project Objective
The primary objective of this project is to leverage YOLOv7's advanced capabilities for real-time object detection. By implementing and fine-tuning this model, we aim to achieve robust and efficient detection of multiple objects in dynamic environments.

### 1.2 Motivation
The motivation behind this project is to harness the power of YOLOv7 to push the boundaries of object detection technology. YOLOv7’s blend of speed and accuracy presents a significant advancement in the field, enabling practical applications where real-time processing is crucial.

### 1.3 Key Features
- High Performance: Achieves state-of-the-art accuracy while maintaining real-time processing speeds.
- Versatile Applications: Suitable for a wide range of use cases, including autonomous vehicles, security systems, and more.
- Robust Detection: Capable of detecting multiple objects in diverse and complex environments.
  
### 1.4 Technologies Used
- YOLOv7 Model: Latest version of the YOLO object detection architecture.
- PyTorch: Deep learning framework used for training and inference.

- Python: Primary programming language for implementation.
  


### 1.5 Utility
YOLOv7 serves as an essential tool for developers and researchers focused on real-time object detection. Its application can extend to:

- Enhancing safety and efficiency in autonomous systems.
- Providing robust security and surveillance solutions.
- Enabling innovative AI-driven applications in various industries.


## 2. Inference





## **1. Setting up Dependencies**
### 1.1 Mounting Google Drive
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### 1.2 Cloning the repo and setting up dependencies
```python
%%bash
cd /content/gdrive/MyDrive
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://raw.githubusercontent.com/WongKinYiu/yolov7/u5/requirements.txt
pip install -r requirements.txt
```
```python
import os
import sys
sys.path.append('/content/gdrive/MyDrive/yolov7')
cd /content/gdrive/MyDrive/yolov7
if not os.path.isdir("/content/gdrive/MyDrive/yolov7/weights"):
  os.makedirs("/content/gdrive/MyDrive/yolov7/weights")
```
### 1.3Getting YOLOv7 Models
```python
%%bash
wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
wget -P /content/gdrive/MyDrive/yolov7/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```




### 1.4 Helper code for inference.
```python
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
```

### **1.5 Configuration Parameters**
```python
classes_to_filter = ['train'] #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


opt  = {

    "weights": "weights/yolov7.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None

}
```
## **2. Inference on single image**
```python
# Give path of source image
source_image_path = '/content/gdrive/MyDrive/yolov7/inference/images/horses.jpg'


with torch.no_grad():
  weights, imgsz = opt['weights'], opt['img-size']
  set_logging()
  device = select_device(opt['device'])
  half = device.type != 'cpu'
  model = attempt_load(weights, map_location=device)  # load FP32 model
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(imgsz, s=stride)  # check img_size
  if half:
    model.half()

  names = model.module.names if hasattr(model, 'module') else model.names
  colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
  if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

  img0 = cv2.imread(source_image_path)
  img = letterbox(img0, imgsz, stride=stride)[0]
  img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
  img = np.ascontiguousarray(img)
  img = torch.from_numpy(img).to(device)
  img = img.half() if half else img.float()  # uint8 to fp16/32
  img /= 255.0  # 0 - 255 to 0.0 - 1.0
  if img.ndimension() == 3:
    img = img.unsqueeze(0)

  # Inference
  t1 = time_synchronized()
  pred = model(img, augment= False)[0]

  # Apply NMS
  classes = None
  if opt['classes']:
    classes = []
    for class_name in opt['classes']:

      classes.append(names.index(class_name))

  if classes:

    classes = [i for i in range(len(names)) if i not in classes]


  pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= [17], agnostic= False)
  t2 = time_synchronized()
  for i, det in enumerate(pred):
    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
    if len(det):
      det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

      for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

      for *xyxy, conf, cls in reversed(det):

        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)


```
```python
from google.colab.patches import cv2_imshow
cv2_imshow(img0)
```
![Alt text](Evaluation.png)


## Citations

This project leverages the groundbreaking advancements in the YOLOv7 model. Special thanks to the researchers behind YOLOv7 and to Augmented Startups for their comprehensive tutorials and resources.

- **YOLOv7 Research Paper**: Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." arXiv preprint arXiv:2207.02696 (2022). [YOLOv7 Paper](https://arxiv.org/abs/2207.02696)
  
- **Augmented Startups**: For their detailed guides and tutorials on implementing YOLOv7. [Augmented Startups YOLOv7 Guide](https://augmentedstartups.info/YOLOv7GetStarted)


