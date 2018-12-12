
from graph_mscoco import *
import numpy as np
import os

MODEL_PATH = os.path.join('InPainting', 'model_mscoco')

def masking(img, mask):
    mask = mask.astype(np.bool)
    mask = np.dstack([mask, mask, mask])
    return (True ^ mask) * np.array(img)

def remove_masked(raw_img, mask):

    height, width = raw_img.shape[:2]

    tf.reset_default_graph()

    with tf.Session() as sess:

        is_train = tf.placeholder(tf.bool)
        images_tf = tf.placeholder(tf.float32, shape=[1, height, width, 3], name="images")

        model = Model()
        reconstruction = model.build_reconstruction(images_tf, is_train)

        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        img = np.array(raw_img) / 255.

        masked_input = masking(img, mask)
        
        shape3d = np.array(masked_input).shape
        
        model_input = np.array( masked_input ).reshape(1, shape3d[0], shape3d[1], shape3d[2])
        
        model_output = sess.run(reconstruction, feed_dict={images_tf: model_input, is_train: False})
    
    result_img = np.array(model_output)[0,:,:,:].astype(float)* 255
    
    return result_img
