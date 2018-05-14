import os
os.environ["KERAS_BACKEND"] = 'tensorflow'
import tensorflow as tf
from keras.layers import UpSampling2D
import numpy as np
import math
IM_SIZE = 128

# use 6 laplacian levels
N_LAP_LEVELS = 6

# precalc filter for faster backprop
FILTER = np.asarray([[.05, .25, .4, .25, .05]])
FILTER = np.dot(FILTER.T, FILTER).reshape(5, 5, 1).astype(np.float32)
DN_SIGMAS = np.asarray([0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782])

GAUSS_FILTER = tf.constant(FILTER, shape=(5, 5, 1, 1), dtype=tf.float32)


def nlp_loss(y_true, y_pred):
    """
    Turns the NLP measure into a loss function from
    eero's lab

    :param y_true:
    :param y_pred:
    :return:
    """

    # apply poser function
    gamma = 1/2.6
    y_true_pwr = tf.pow(y_true, gamma)
    y_pred_pwr = tf.pow(y_pred, gamma)

    # filters come from the paper
    DN_filts = tf.constant(_dn_filters())

    # calculate the laplacian levels for each image
    y_ori, lap_ori = _nlp(y_true_pwr, dn_filts=DN_filts)
    y_dist, lap_dist = _nlp(y_pred_pwr, dn_filts=DN_filts)

    # calculate MSE for each laplacian level
    rr_aux = []
    for y_ori_i, y_dist_i in zip(y_ori, y_dist):
        rr_aux.append(tf.reduce_mean(tf.squared_difference(y_ori_i, y_dist_i)))

    # calculate overall mse as loss
    DMOS_Lap_dn2 = tf.reduce_mean(rr_aux)
    return DMOS_Lap_dn2


def _nlp(img, dn_filts):
    """
    Create the pyramid and normalize it
    :param img:
    :param dn_filts:
    :return:
    """
    # generate a list of tensors representing the down and upsampling
    # operations from the laplacian pyramid algorithm
    lap_pyramid = _laplacian_pyramid_s(img)

    dn_dom = []
    for level_i in range(N_LAP_LEVELS):
        # get pointers to each tensor
        dn_filter = dn_filts[level_i]
        dn_sigma = DN_SIGMAS[level_i]
        pyramid_level = lap_pyramid[level_i]

        # do a non-strided convolution to calculate the normalizing factor
        a2 = tf.nn.conv2d(tf.abs(pyramid_level), dn_filter, padding='SAME', strides=(1, 1, 1, 1))

        # elementwise divide between each filter's sigma and the result of the convolution from above
        # this normalizes the laplacian pyramid
        normalizing_factor = (dn_sigma + a2)
        dn_dom.append(pyramid_level / normalizing_factor)

    return dn_dom, lap_pyramid


def _laplacian_pyramid_s(im):

    # this is the filter we'll convolve over for the pyramid
    down_filter = tf.constant(FILTER, shape=(5, 5, 1, 1))

    lap_pyramid = []
    imgs = []
    j = im
    for level_i in range(MAX_LEVEL - 1):
        # Split the image into lo and hi frequency components

        # use strided convolution for downsampling
        # then do a deconv operation to upsample
        down_sampled = tf.nn.conv2d(j, down_filter, strides=[1, 2, 2, 1], padding='SAME')

        # deconv through upsampling + deconv
        up_i = UpSampling2D(trainable=False)(down_sampled)
        up_i = tf.nn.conv2d(up_i, down_filter, strides=[1, 1, 1, 1], padding='SAME')

        # in each level, store difference between image and upsampled low pass version
        # bc tf needs arrays of the same size, we'll place the result in the NW corner
        # so that the rest of the matrix has zeros
        hi_lo_diff = j - up_i
        lap_pyramid.append(hi_lo_diff)
        imgs.append(j)
        j = down_sampled

    # the coarest level contains the residual low pass image
    lap_pyramid.append(j)
    return lap_pyramid


def _dn_filters():
    """
    These are from the eero paper / matlab code
    :return:
    """

    filter_1 = np.asarray([[0,      0,      0,      0,      0],
                           [0,      0,      0.1011, 0,      0],
                           [0,      0.1493, 0,      0.1460, 0.0072],
                           [0,      0,      0.1015, 0,      0],
                           [0,      0,      0,      0,      0]
                           ])

    filter_2 = np.asarray([[0,      0,      0,      0,      0],
                           [0,      0,      0.0757, 0,      0],
                           [0,      0.1986, 0,      0.1846, 0],
                           [0,      0,      0.0837, 0,      0],
                           [0,      0,      0,      0,      0]
                           ])

    filter_3 = np.asarray([[0,      0,      0,      0,      0],
                           [0,      0,      0.0477, 0,      0],
                           [0,      0.2138, 0,      0.2243, 0],
                           [0,      0,      0.0467, 0,      0],
                           [0,      0,      0,      0,      0]
                           ])

    filter_4 = np.asarray([[0,      0,      0,      0,      0],
                           [0,      0,      0,      0,      0],
                           [0,      0.2503, 0,      0.2616, 0],
                           [0,      0,      0,      0,      0],
                           [0,      0,      0,      0,      0]
                           ])

    filter_5 = np.asarray([[0,      0,      0,      0,      0],
                           [0,      0,      0,      0,      0],
                           [0,      0.2598, 0,      0.2552, 0],
                           [0,      0,      0,      0,      0],
                           [0,      0,      0,      0,      0]
                           ])

    filter_6 = np.asarray([[0,      0,      0,      0,      0],
                           [0,      0,      0,      0,      0],
                           [0,      0.2215, 0,      0.0717, 0],
                           [0,      0,      0,      0,      0],
                           [0,      0,      0,      0,      0]
                           ])

    # make tensorflow dim checker happy
    # filter w x filter h x nb_channels x nb_filters
    filter_1 = filter_1.reshape(5, 5, 1, 1).astype(np.float32)
    filter_2 = filter_2.reshape(5, 5, 1, 1).astype(np.float32)
    filter_3 = filter_3.reshape(5, 5, 1, 1).astype(np.float32)
    filter_4 = filter_4.reshape(5, 5, 1, 1).astype(np.float32)
    filter_5 = filter_5.reshape(5, 5, 1, 1).astype(np.float32)
    filter_6 = filter_6.reshape(5, 5, 1, 1).astype(np.float32)

    return np.asarray([filter_1, filter_2, filter_3, filter_4, filter_5, filter_6])


def max_floor_level(im_width, im_height):
    r, c = im_width, im_height

    # calculate max level possible
    nlev = math.floor(math.log(min(r, c)) / math.log(2.0))
    return nlev

MAX_LEVEL = max_floor_level(IM_SIZE, IM_SIZE)


