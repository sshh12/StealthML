from glob import glob
import os
import numpy as np
import cv2
from graph_mscoco import *

pen_size = 3
img_idx = 0
drawing = False
ix, iy = -1, -1
vis_size = 320
blank_size = 20

def masking(img):
    mask = (np.array(img[:,:,0]) == 0) & (np.array(img[:,:,1]) == 0) & (np.array(img[:,:,2]) == 0)
    mask = np.dstack([mask,mask,mask]);
    return (True ^ mask) * np.array(img)


img = cv2.imread('test.bmp') / 255.

sess = tf.InteractiveSession()
pretrained_model_path = 'model_mscoco'
is_train = tf.placeholder( tf.bool )
images_tf = tf.placeholder( tf.float32, shape=[1, vis_size, vis_size, 3], name="images")
model = Model()
reconstruction_ori = model.build_reconstruction(images_tf, is_train)
saver = tf.train.Saver(max_to_keep=100)
saver.restore( sess, pretrained_model_path )

img[20:60, 20:60, :] = [0, 0, 0]

masked_input = masking(img)
masked_input = masked_input[:,:,[2,1,0]]
shape3d = np.array( masked_input ).shape
model_input = np.array( masked_input ).reshape(1, shape3d[0], shape3d[1], shape3d[2])
model_output = sess.run(reconstruction_ori,feed_dict={images_tf: model_input, is_train: False})
recon_img = np.array(model_output)[0,:,:,:].astype(float)
res = (recon_img[:,:,:]) * 255

import PIL.Image
PIL.Image.fromarray(res.astype(np.uint8)).show()


