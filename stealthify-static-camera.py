"""
Stealthify a static video.

Uses image segmentation to identify targets
but skips the neural inpainting step since
background already known (static = camera is fixed).
"""


# Use lib as if they were installed
import sys
sys.path.insert(0, "./InPainting")
sys.path.insert(0, "./MaskRCNN")

# Utils
import skimage.transform
import cv2
import numpy as np

import find_mask


def inflate_mask(mask):
    """Increase the size of the masked region"""
    kernel = np.ones((12, 12), np.uint8)
    return cv2.dilate(mask, kernel, 1)


def background_inpaint(image, mask, background):
    """Use the background image to inpaint the region"""
    mask_idx = np.where(mask == 1)
    result = np.array(image)
    result[mask_idx] = background[mask_idx]
    return result


def main(thing, video_device=0):

    background = None
    background_set = False

    cap = cv2.VideoCapture(video_device)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # warm up the camera a bit...
    for i in range(10):
        ret, frame = cap.read()

    while True:

        ret, frame = cap.read()

        image = frame[:, :, [2, 1, 0]]
        image = skimage.transform.resize(image, (480, 640)) * 255

        # use the first frame as the background
        if not background_set:
            background = image
            background_set = True
            continue

        class_mask = find_mask.find(image, thing)
        class_mask = inflate_mask(class_mask)
        
        result = background_inpaint(image, class_mask, background)

        cv2.imshow(f'No {thing}s here...', result[:, :, [2, 1, 0]] / 255.)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main('person')
