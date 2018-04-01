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

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

from config import Config

class InferenceConfig(Config):
    BACKBONE_SHAPES                = [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16]]
    BACKBONE_STRIDES               = [4, 8, 16, 32, 64]
    BATCH_SIZE                     = 1
    BBOX_STD_DEV                   = [0.1, 0.1, 0.2, 0.2]
    DETECTION_MAX_INSTANCES        = 100
    DETECTION_MIN_CONFIDENCE       = 0.5
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
config.display()


# ## Create Model and Load Trained Weights


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
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


# ## Run Object Detection

# In[10]:


# Load a random image from the images folder
#filename = r'C:\Users\Shriv\Desktop\Mask_RCNN\images\12283150_12d37e6389_z.jpg'
#image = skimage.io.imread(filename)
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display
from visualize import *
import utils

def display_instances2(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    # ax.imshow(masked_image.astype(np.uint8))
    return masked_image.astype(np.uint8)
import numpy as np
import cv2
from skimage.transform import resize
import time
from scipy.misc import imshow
while True:

    cap = cv2.VideoCapture('http://10.0.0.11:4747/mjpegfeed?640x480')

    ret, frame = cap.read()

    if frame is None:
        continue

    f2 = resize(frame, (1024, 1024, 3))

    image = frame

    # Run detection
    results = model.detect([image], verbose=1)
    
    # Visualize results
    #fig, ax = plt.subplots()
    #
    r = results[0]
    x = display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])

    #fig.canvas.draw()
    #buf = fig.canvas.tostring_rgb()
    #ncols, nrows = fig.canvas.get_width_height()
    #img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    #cv2.imshow('Video', x)
    
    
