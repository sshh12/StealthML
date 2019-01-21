# StealthML

Using image segmentation and in-painting to tinker with images.

## How it works

```bash
# Remove all zebras from test.jpg and display results
$ python stealthify.py --file test.jpg --class_name zebra --display
```

1. The image is run through Mask RCNN and a mask is generated for the class name specified
2. The masked region is then removed from the original image and the in-painting model fills in the hole

## Models

### Segmentation

This is done with the Mask RCNN from [matterport / Mask_RCNN](https://github.com/matterport/Mask_RCNN).

[Download Weights](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)

```python
# Supported Classes
['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
```

### In-Painting

Currently using cGANs from [adamstseng / general-deep-image-completion](https://github.com/adamstseng/general-deep-image-completion).

[Download Weights](https://drive.google.com/file/d/0BwBvCjzIsl2vV3FvZUd0VjdxZE0/view?usp=sharing)

## Results

If you ignore the *suddle* lighting deviations, the low FPS, the frame with the umbrella visable, and the fact that this was taking advantage of a static background rather than actual in-painting...this worked pretty well.

#### Cloaking Umbrella POC

![bad gif](https://user-images.githubusercontent.com/6625384/51450883-87f9a080-1d00-11e9-98d6-4f5983bdb5b8.gif)
