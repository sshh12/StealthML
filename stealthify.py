import sys
sys.path.insert(0, "./InPainting")
sys.path.insert(0, "./MaskRCNN")

import skimage.io
import skimage.transform
import cv2
import numpy as np
import PIL.Image

import find_mask
import inpaint


def main(fn, thing):

    image = skimage.io.imread(fn)
    # image = skimage.transform.resize(image, (320, 320)) * 255

    class_mask = find_mask.find(image, thing)
    
    kernel = np.ones((16, 16), np.uint8)
    class_mask = cv2.dilate(class_mask, kernel, 1)
    
    result = inpaint.remove_masked(image, class_mask)

    PIL.Image.fromarray(result.astype(np.uint8)).show()


if __name__ == "__main__":
    main('test.jpg', 'giraffe')
