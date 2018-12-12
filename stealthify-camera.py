# Use lib as if they were installed
import sys
sys.path.insert(0, "./InPainting")
sys.path.insert(0, "./MaskRCNN")

# Utils
import skimage.transform
import cv2
import numpy as np

import find_mask
import inpaint


def inflate_mask(mask):
    """Increase the size of the masked region"""
    kernel = np.ones((12, 12), np.uint8)
    return cv2.dilate(mask, kernel, 1)


def main(thing, video_device=0):

    cap = cv2.VideoCapture(video_device)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:

        ret, frame = cap.read()

        image = frame[:, :, [2, 1, 0]]
        image = skimage.transform.resize(image, (400, 600)) * 255

        class_mask = find_mask.find(image, thing)
        class_mask = inflate_mask(class_mask)
        
        result = inpaint.remove_masked(image, class_mask)

        cv2.imshow(f'No {thing}s here...', result[:, :, [2, 1, 0]] / 255.)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main('spoon')
