import os
os.environ["KERAS_BACKEND"] = 'tensorflow'
import keras.backend as K
from keras.layers import ZeroPadding2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import h5py
"""
Appends the requested parts of a VGG net to the computational graph.
It's parametrized so it can break the network into any arbitrary layer
"""

IM_SIZE = 128
LAYER_BREAK = 'b2_c2'

# download VGG weights from here:
# https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
VGG_WEIGHTS_DIR = None


def vgg_loss(break_layer_name, im_width, weights_dir):
    """
    Wrapper function to allow configuring the layer to break on
    and the weights path

    :param break_layer_name:
    :param im_width:
    :param weights_dir:
    :return:
    """
    global LAYER_BREAK
    global IM_SIZE
    global VGG_WEIGHTS_DIR

    # transfer the constants to the global scope
    LAYER_BREAK = break_layer_name
    IM_SIZE = im_width
    VGG_WEIGHTS_DIR = weights_dir

    # return a pointer to the loss fx
    return _vgg_loss


# ---------------------------------
# UTILS
# ---------------------------------


def _vgg_loss(y_true, y_pred):
    """
    VGG loss works by running images through n layers in the VGG
    and measuring the MSE distance from the output params

    Run each y and y_hat through the VGG net and calculate
    the distance loss from the output of:
    Block group 2, conv layer 2

    :param y_true:
    :param y_pred:
    :return:
    """

    # make our images 3 channels so we can run through VGG net
    y_true_3_channel = tf.concat([y_true, y_true, y_true], -1)
    y_pred_3_channel = tf.concat([y_pred, y_pred, y_pred], -1)

    # run images through vgg net to extract feature maps
    # rescale_factor from SRGAN paper
    rescale_factor = (1 / 12.75)
    vgg_y_true = _root_vgg_tf(y_true_3_channel, LAYER_BREAK) * rescale_factor
    vgg_y_pred = _root_vgg_tf(y_pred_3_channel, LAYER_BREAK) * rescale_factor

    # return the mse between the feature maps
    return _calc_mse(vgg_y_true, vgg_y_pred)


def _calc_mse(vgg_y_true, vgg_y_pred):
    """
    Helper to measure MSE in feature map space
    :param vgg_y_true:
    :param vgg_y_pred:
    :return:
    """
    return tf.reduce_mean(tf.squared_difference(vgg_y_true, vgg_y_pred))


def Conv_w_weights(input, map_size, filter_w, filter_h, weight_i, activation='relu'):
    """
    Wrapper around Conv layer that loads the weights into the layer
    :param input_tensor:
    :param map_size:
    :param filter_w:
    :param filter_h:
    :param weight_i:
    :param activation:
    :return:
    """
    w, b = _get_weights_for_layer(weight_i)

    # drop into tf for the conv layer... wasn't working on keras, so we
    # preload this layer with weights
    conv_kernel_1 = tf.nn.conv2d(input, w, [1, 2, 2, 1], padding='VALID')
    bias_layer_1 = tf.nn.bias_add(conv_kernel_1, b)

    return bias_layer_1


def _get_weights_for_layer(i):
    """
    Helper function to load the VGG weights for the ith layer
    :param i:
    :return:
    """
    # ensure we have the weights for VGG
    try:

        weights = np.array(h5py.File(VGG_WEIGHTS_DIR, mode='r')['layer_{}'.format(i)]['param_0'], dtype=np.float32)
        biases = np.array(h5py.File(VGG_WEIGHTS_DIR, mode='r')['layer_{}'.format(i)]['param_1'], dtype=np.float32)
        weights = np.transpose(weights, [3, 2, 1, 0])
        return weights, biases

    except Exception as e:
        print('Download from: http://bit.ly/vgg16_weights')
        print(e)
        exit(1)


def _root_vgg_tf(y, end_layer_name):
    """
    Push the tensor through the VGG net
    The net is broken off at the end_layer_name
    Available values are:
    [b1_c1, b1_c2, b2_c1, b2_c2, b3_c1, b3_c2, b3_c3,
    b4_c1, b4_c2, b4_c3, b5_c1, b5_c2, b5_c3]

    b = Block
    c = Conv layer

    :param y:
    :param end_layer_name:
    :return:
    """
    # -----------------------------
    # BLOCK 1
    out = ZeroPadding2D((1, 1),input_shape=(None, IM_SIZE, IM_SIZE, 3), trainable=False)(y)
    out = Conv_w_weights(out, 64, 3, 3, weight_i=1)
    if end_layer_name == 'b1_c1': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 64, 3, 3, weight_i=3)
    if end_layer_name == 'b1_c2': return out

    out = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(out)

    # -----------------------------
    # BLOCK 2
    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 128, 3, 3, weight_i=6)
    if end_layer_name == 'b2_c1': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 128, 3, 3, weight_i=8)
    if end_layer_name == 'b2_c2': return out

    out = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(out)

    # -----------------------------
    # BLOCK 3
    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 256, 3, 3, weight_i=11)
    if end_layer_name == 'b3_c1': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 256, 3, 3, weight_i=13)
    if end_layer_name == 'b3_c2': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 256, 3, 3, weight_i=15)
    if end_layer_name == 'b3_c3': return out

    out = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(out)

    # -----------------------------
    # BLOCK 4
    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 512, 3, 3, weight_i=18)
    if end_layer_name == 'b4_c1': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 512, 3, 3, weight_i=20)
    if end_layer_name == 'b4_c2': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 512, 3, 3, weight_i=22)
    if end_layer_name == 'b4_c3': return out

    out = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(out)

    # -----------------------------
    # BLOCK 5
    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 512, 3, 3, weight_i=25)
    if end_layer_name == 'b5_c1': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 512, 3, 3, weight_i=27)
    if end_layer_name == 'b5_c2': return out

    out = ZeroPadding2D((1, 1), trainable=False)(out)
    out = Conv_w_weights(out, 512, 3, 3, weight_i=29)
    if end_layer_name == 'b5_c3': return out

    out = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(out)

# --------------------------
# TESTING LOCALLY
# --------------------------
if __name__ == '__main__':
    from skimage.io import imsave, imread

    sess = tf.Session()
    K.set_session(sess)

    IM_SIZE = 128
    x = tf.placeholder(tf.float32, shape=(None, IM_SIZE, IM_SIZE, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, IM_SIZE, IM_SIZE, 1), name='y')

    #X = imread('/Users/waf/Desktop/messi5.jpg')[0:IM_SIZE, 0:IM_SIZE, 1].reshape(1, IM_SIZE, IM_SIZE, 1)/255.0
    X = imread('/Users/waf/Desktop/b.jpg')[0:IM_SIZE, 0:IM_SIZE, 0].reshape(1, IM_SIZE, IM_SIZE, 1)/255.0
    Y = imread('/Users/waf/Desktop/d.png')[0:IM_SIZE, 0:IM_SIZE, 0].reshape(1, IM_SIZE, IM_SIZE, 1)/255.0

    # init vars
    ans = vgg_loss_block2_conv2(x, y)

    with tf.device('cpu:0'):
        mse = sess.run(ans, feed_dict={x: X, y: Y})
        # for i, im in enumerate(lap_dom):
        #     s = im.shape[1]
        #     imsave('/Users/waf/Desktop/dd{}.png'.format(i), im.reshape(s, s))
        print(mse)



