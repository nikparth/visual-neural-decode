import tensorflow as tf
import numpy as np

FILTER = np.asarray([[.05, .25, .4, .25, .05]])
FILTER = np.dot(FILTER.T, FILTER).reshape(5, 5, 1).astype(np.float32)
GAUSS_FILTER = tf.constant(FILTER, shape=(5, 5, 1, 1), dtype=tf.float32)


# ---------------------
# LAP CONSTANT LEVEL LOSS
# ---------------------
def lap_rnn_loss_mse(y_true, y_pred, nb_laplace_levels, filter_name='sobel'):
    """
    :param y_true:
    :param y_pred:
    :param nb_laplace_levels:
    :param filters: Possible filters 'gaussian', 'sobel', 'vgg (layer 1)'
    :return:
    """
    # select the appropriate filters for the network
    filters = GAUSS_FILTER
    if filter_name == 'sobel':
        filters = sobel_filters()

    y_true_feat_maps = calc_img_residual_levels(y_true, nb_laplace_levels, filters)
    y_pred_feat_maps = calc_img_residual_levels(y_pred, nb_laplace_levels, filters)

    means = []
    for y_true_map, y_pred_map in zip(y_true_feat_maps, y_pred_feat_maps):
        means.append(tf.reduce_mean(tf.squared_difference(y_true_map, y_pred_map)))

    return tf.reduce_mean(means)


def lap_rnn_loss_mae(y_true, y_pred, nb_laplace_levels):

    y_true_feat_maps = calc_img_residual_levels(y_true, nb_laplace_levels)
    y_pred_feat_maps = calc_img_residual_levels(y_pred, nb_laplace_levels)

    means = []
    for y_true_map, y_pred_map in zip(y_true_feat_maps, y_pred_feat_maps):
        means.append(tf.reduce_mean(tf.abs(tf.subtract(y_true_map, y_pred_map))))

    return tf.reduce_mean(means)


def calc_img_residual_levels(images, nb_laplace_levels, filters):
    prior_image = images
    residual_imgs = []

    for i in range(nb_laplace_levels):

        # conv to blur
        blurred_img = tf.nn.conv2d(prior_image, filters, strides=[1, 1, 1, 1], padding='SAME')
        residual_img = images - blurred_img
        residual_imgs.append(residual_img)

        # move on to next level
        prior_image = images - residual_img

    return np.asarray(residual_imgs)


def sobel_filters():
    left = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    right = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    filter = np.asarray([left, right]).reshape(3, 3, 2, 1).astype(np.float32)
    return tf.constant(filter, shape=(3, 3, 2, 1), dtype=tf.float32)
