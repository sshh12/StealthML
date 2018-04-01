import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import utils
import model as modellib
import visualize

from config import Config

ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(Config):
    BACKBONE_SHAPES                = [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16]]
    BACKBONE_STRIDES               = [4, 8, 16, 32, 64]
    BATCH_SIZE                     = 1
    BBOX_STD_DEV                   = [0.1, 0.1, 0.2, 0.2]
    DETECTION_MAX_INSTANCES        = 80
    DETECTION_MIN_CONFIDENCE       = 0.85
    DETECTION_NMS_THRESHOLD        = 0.3
    GPU_COUNT                      = 1
    IMAGES_PER_GPU                 = 1
    IMAGE_MAX_DIM                  = 1024
    IMAGE_MIN_DIM                  = 800
    IMAGE_PADDING                  = True
    IMAGE_SHAPE                    = [1024, 1024, 3]
    LEARNING_MOMENTUM              = 0.9
    LEARNING_RATE                  = 0.002
    MASK_POOL_SIZE                 = 14
    MASK_SHAPE                     = [28, 28]
    MAX_GT_INSTANCES               = 100
    MEAN_PIXEL                     = [123.7, 116.8, 103.9]
    MINI_MASK_SHAPE                = (56, 56)
    NAME                           = "InfConfig"
    NUM_CLASSES                    = 81
    POOL_SIZE                      = 7
    POST_NMS_ROIS_INFERENCE        = 1000
    POST_NMS_ROIS_TRAINING         = 2000
    ROI_POSITIVE_RATIO             = 0.33
    RPN_ANCHOR_RATIOS              = [0.5, 1, 2]
    RPN_ANCHOR_SCALES              = (32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE              = 2
    RPN_BBOX_STD_DEV               = [0.1, 0.1, 0.2, 0.2]
    RPN_TRAIN_ANCHORS_PER_IMAGE    = 256
    STEPS_PER_EPOCH                = 1000
    TRAIN_ROIS_PER_IMAGE           = 128
    USE_MINI_MASK                  = True
    USE_RPN_ROIS                   = True
    VALIDATION_STEPS               = 50
    WEIGHT_DECAY                   = 0.0001
    
config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

## END SETUP ##

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

import cv2

cap = cv2.VideoCapture("http://10.0.0.11:4747/mjpegfeed?960x720")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

colors = visualize.random_colors(len(class_names))

def bbox(img):
    
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

while True:

    for i in range(25): # B/c buffer

        ret, frame = cap.read()

    image = frame

    results = model.detect([image], verbose=0)
    r = results[0]

    for i, class_id in enumerate(r['class_ids']):

        visualize.apply_mask(image, r['masks'][:, :, i], colors[class_id])

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    for i, class_id in enumerate(r['class_ids']):

        y, y2, x, x2 = bbox(r['masks'][:, :, i])

        draw.text(((x + x2) / 2, (y + y2) / 2), class_names[class_id] + " " + str(round(r['scores'][i], 2)), (0, 255, 0))

    image = np.array(img)

    cv2.imshow('Live Seg', image)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
    
    
