import os
os.environ["KERAS_BACKEND"] = 'tensorflow'

from skimage.io import imread
import keras.backend as k
from matplotlib import pyplot as plt
from nlp_loss import nlp_loss
from lp_rnn_loss import calc_img_residual_levels
IM_SIZE = 128


if __name__ == '__main__':
    import tensorflow as tf
    import os
    from skimage.io import imsave
    import numpy as np

    sess = tf.Session()
    k.set_session(sess)
    IM_SIZE = 128
    x = tf.placeholder(tf.float32, shape=(None, IM_SIZE, IM_SIZE, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, IM_SIZE, IM_SIZE, 1), name='y')
    X = imread('/Users/waf/Desktop/face.jpg')[0:IM_SIZE, 128*2:].reshape(1, IM_SIZE, IM_SIZE, 1)
    # X = np.tile(X, reps=(1, 1, 1, 1))
    Y = imread('/Users/waf/Desktop/face.jpg')[0:IM_SIZE, 128:128*2].reshape(1, IM_SIZE, IM_SIZE, 1)
    # Y = np.tile(Y, reps=(1, 1, 1, 1))

    #X = crop_image_batch(X, IM_SIZE).reshape(1, IM_SIZE, IM_SIZE, 1)

    p_out = calc_img_residual_levels(x, 50)
    residual_imgs = sess.run(p_out, feed_dict={x: X})
    imsave('/Users/waf/Desktop/lap_test/orig.jpg', X.reshape(X.shape[1], X.shape[1]), cmap='jet')
    for i, dn in enumerate(residual_imgs):
        # save normed img
        plt.imsave('/Users/waf/Desktop/lap_test/diff_dn_{}.jpg'.format(i), dn.reshape(dn.shape[1], dn.shape[1]))

    exit(-1)

    #Y = imread('/Users/waf/Desktop/a.png')[0:IM_SIZE, 0:IM_SIZE, 0].reshape(1, IM_SIZE, IM_SIZE, 1)/255.0

    # f = new_var('test', 'v')

    # init vars
    ans = nlp_loss(x, y)

    with tf.device('/cpu:0'):
        # sess.run(tf.global_variables_initializer())
        lap_dom = sess.run(ans, feed_dict={x: X, y: Y})

        # sess.run(f.assign_add(200.0))
        # for i, im in enumerate(lap_dom):
        #     s = im.shape[1]
        #     imsave('/Users/waf/Desktop/dd{}.png'.format(i), im.reshape(s, s))
        print(lap_dom)
        lap_dom = sess.run(ans, feed_dict={x: X, y: Y})
        print(lap_dom)



