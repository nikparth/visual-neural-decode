import tensorflow as tf
from keras.layers import UpSampling2D


def cae_nn(input_x, drop_rate, phase_train):

    # generate CAE
    encoder = encoder_nn(nn_input=input_x, drop_rate=drop_rate, phase_train=phase_train)
    cae = decoder_nn(encoder, drop_rate=drop_rate, phase_train=phase_train)

    return cae


def encoder_nn(nn_input, drop_rate, phase_train):
    with tf.variable_scope('encoder_nn') as scope:
        # layer 1
        enc_1 = tf.layers.conv2d(inputs=nn_input, filters=64, kernel_size=7, strides=[2, 2], padding='same')
        enc_1 = tf.layers.batch_normalization(inputs=enc_1, training=phase_train)
        enc_1 = tf.nn.relu(features=enc_1)

        # layer 2
        enc_2 = tf.layers.conv2d(inputs=enc_1, filters=128, kernel_size=5, strides=[2, 2], padding='same')
        enc_2 = tf.layers.batch_normalization(inputs=enc_2, training=phase_train)
        enc_2 = tf.nn.relu(features=enc_2)
        enc_2 = tf.layers.dropout(inputs=enc_2, rate=drop_rate)

        # layer 3
        enc_3 = tf.layers.conv2d(inputs=enc_2, filters=256, kernel_size=3, strides=[2, 2], padding='same')
        enc_3 = tf.layers.batch_normalization(inputs=enc_3, training=phase_train)
        enc_3 = tf.nn.relu(features=enc_3)
        enc_3 = tf.layers.dropout(inputs=enc_3, rate=drop_rate)

        # layer 4
        enc_4 = tf.layers.conv2d(inputs=enc_3, filters=256, kernel_size=3, strides=[2, 2], padding='same')
        enc_4 = tf.layers.batch_normalization(inputs=enc_4, training=phase_train)
        enc_4 = tf.nn.relu(features=enc_4)
        enc_4 = tf.layers.dropout(inputs=enc_4, rate=drop_rate, name='encoder_out')

    return enc_4


def decoder_nn(enc_n, drop_rate, phase_train):
    with tf.variable_scope('decoder_nn') as scope:

        # layer 1
        dec_1 = UpSampling2D(size=(2, 2), data_format='channels_last')(enc_n)
        dec_1 = tf.layers.conv2d(inputs=dec_1, filters=256, kernel_size=3, strides=[1, 1], padding='same')
        dec_1 = tf.layers.batch_normalization(inputs=dec_1, training=phase_train)
        dec_1 = tf.nn.relu(features=dec_1)
        dec_1 = tf.layers.dropout(inputs=dec_1, rate=drop_rate)

        # layer 2
        dec_2 = UpSampling2D(size=(2, 2), data_format='channels_last')(dec_1)
        dec_2 = tf.layers.conv2d(inputs=dec_2, filters=128, kernel_size=3, strides=[1, 1], padding='same')
        dec_2 = tf.layers.batch_normalization(inputs=dec_2, training=phase_train)
        dec_2 = tf.nn.relu(features=dec_2)
        dec_2 = tf.layers.dropout(inputs=dec_2, rate=drop_rate)

        # layer 3
        dec_3 = UpSampling2D(size=(2, 2), data_format='channels_last')(dec_2)
        dec_3 = tf.layers.conv2d(inputs=dec_3, filters=64, kernel_size=5, strides=[1, 1], padding='same')
        dec_3 = tf.layers.batch_normalization(inputs=dec_3, training=phase_train)
        dec_3 = tf.nn.relu(features=dec_3)
        dec_3 = tf.layers.dropout(inputs=dec_3, rate=drop_rate)

        # layer 4
        dec_4 = UpSampling2D(size=(2, 2), data_format='channels_last')(dec_3)
        dec_4 = tf.layers.conv2d(inputs=dec_4, filters=1, kernel_size=7, strides=[1, 1], padding='same')

        # out layer
        dec_4 = tf.layers.batch_normalization(inputs=dec_4, name='cae_out', training=phase_train)
        tf.add_to_collection('predict_opt', dec_4)

    return dec_4
