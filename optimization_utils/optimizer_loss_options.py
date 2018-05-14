import tensorflow as tf
from . import perceptual_losses


def choose_loss(loss_name, y_true, y_pred, loss_weight_ph, loss_opts=None):
    if 'mse_split_nlp'in loss_name:
        return 5.0 * perceptual_losses.nlp_loss(y_true, y_pred) + tf.reduce_mean(tf.squared_difference(y_pred, y_true))

    if 'nlp_vgg'in loss_name:
        vgg_layer_name = loss_name.split('vgg_')[-1]
        vgg_loss = perceptual_losses.vgg_loss(vgg_layer_name, loss_opts['im_width'], loss_opts['vgg_weights_dir'])
        return 2.0 * perceptual_losses.nlp_loss(y_true, y_pred) + vgg_loss(y_true, y_pred)

    if 'mse_split_vgg'in loss_name:
        vgg_layer_name = loss_name.split('vgg_')[-1]
        vgg_loss = perceptual_losses.vgg_loss(vgg_layer_name, loss_opts['im_width'], loss_opts['vgg_weights_dir'])
        return tf.reduce_mean(tf.squared_difference(y_pred, y_true)) + vgg_loss(y_true, y_pred)

    if 'reverse_anneal_vgg'in loss_name:
        vgg_layer_name = loss_name.split('vgg_')[-1]
        vgg_loss = perceptual_losses.vgg_loss(vgg_layer_name, loss_opts['im_width'], loss_opts['vgg_weights_dir'])
        return (1-loss_weight_ph) * vgg_loss(y_true, y_pred) + loss_weight_ph * tf.reduce_mean(tf.squared_difference(y_pred, y_true))

    if 'anneal_vgg'in loss_name:
        vgg_layer_name = loss_name.split('vgg_')[-1]
        vgg_loss = perceptual_losses.vgg_loss(vgg_layer_name, loss_opts['im_width'], loss_opts['vgg_weights_dir'])
        return loss_weight_ph * vgg_loss(y_true, y_pred) + (1-loss_weight_ph) * tf.reduce_mean(tf.squared_difference(y_pred, y_true))

    if 'vgg'in loss_name:
        vgg_layer_name = loss_name[4:]
        vgg_loss = perceptual_losses.vgg_loss(vgg_layer_name, loss_opts['im_width'], loss_opts['vgg_weights_dir'])
        return vgg_loss(y_true, y_pred)

    if loss_name == 'mse':
        return tf.reduce_mean(tf.squared_difference(y_pred, y_true))

    if loss_name == 'mae':
        return tf.reduce_mean(tf.abs(y_pred - y_true))

    if 'anneal_nlp'in loss_name:
        return loss_weight_ph * perceptual_losses.nlp_loss(y_true, y_pred) + (1-loss_weight_ph) * tf.reduce_mean(tf.squared_difference(y_pred, y_true))

    if loss_name == 'nlp':
        return perceptual_losses.nlp_loss(y_true, y_pred)

    if loss_name == 'lap_rnn_mse':
        return perceptual_losses.lap_rnn_loss_mse(y_true, y_pred, loss_opts['nb_laplace_levels'])

    if loss_name == 'sigmoid_cross_entropy':
        return tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true, logits=y_pred)




def choose_optimizer(optimizer, lr, loss):
    """
    Default opt is Adam
    :param optimizer:
    :param lr:
    :param loss:
    :return:
    """
    if optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    elif optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)
    else:
        opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return opt

