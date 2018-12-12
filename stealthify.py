# Use lib as if they were installed
import sys
sys.path.insert(0, "./InPainting")
sys.path.insert(0, "./MaskRCNN")

# Utils
import skimage.io
import skimage.transform
import cv2
import numpy as np
import PIL.Image

import find_mask
import inpaint


def inflate_mask(mask):
    """Increase the size of the masked region"""
    kernel = np.ones((16, 16), np.uint8)
    return cv2.dilate(mask, kernel, 1)


def get_image(array):
    """Convert array(h, w, 3) to image object"""
    return PIL.Image.fromarray(array.astype(np.uint8))


def main(fn, thing, display=True, save_fn=None):
    """Remove `thing` from image"""
    image_ary = skimage.io.imread(fn)

    class_mask = find_mask.find(image_ary, thing)
    class_mask = inflate_mask(class_mask)
    
    result = inpaint.remove_masked(image_ary, class_mask)

    image = get_image(result)

    if display:
        image.show()

    if save_fn:
        image.save(save_fn)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="The image file to use", metavar="filename", required=True)
    parser.add_argument("-c", "--class", dest="class_name", choices=find_mask.class_names, type=str,
                        help="The type of object to hide", metavar="class_name", nargs=1, required=True)
    parser.add_argument("-d", "--display", dest="display", action='store_true',
                        help="Display the resulting image")
    parser.add_argument("-s", "--save", dest="save_file", default='output.jpg', nargs='?',
                        help="Where to save the resulting image", metavar="save_file")

    args = parser.parse_args()

    main(args.filename, args.class_name[0], display=args.display, save_fn=args.save_file)
