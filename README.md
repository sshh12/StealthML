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

### In-Painting

Currently using cGANs from [adamstseng / general-deep-image-completion](https://github.com/adamstseng/general-deep-image-completion).

[Download Weights](https://drive.google.com/file/d/0BwBvCjzIsl2vV3FvZUd0VjdxZE0/view?usp=sharing)

## Results

```TODO```