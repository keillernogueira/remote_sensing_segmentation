import math

from networks.layers import _conv_layer, _max_pool_with_argmax, _up_pooling, \
    _variable_with_weight_decay, _variable_on_cpu

import tensorflow as tf


# https://github.com/toimcio/SegNet-tensorflow/blob/master/SegNet.py
def segnet(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features):
    x = tf.reshape(x, shape=[-1, crop, crop, num_input_bands])
    batch_size = tf.shape(x)[0]

    norm1 = tf.nn.local_response_normalization(x, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

    conv1_1 = _conv_layer(norm1, [3, 3, num_input_bands, 64], "conv1_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], "conv1_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    h_pool1, w_pool1 = crop_size, crop_size
    pool1, pool1_index, shape_1 = _max_pool_with_argmax(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool1', pad='SAME')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], "conv2_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], "conv2_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    h_pool2, w_pool2 = math.ceil(h_pool1 / float(2)), math.ceil(w_pool1 / float(2))
    pool2, pool2_index, shape_2 = _max_pool_with_argmax(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2', pad='SAME')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], "conv3_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], "conv3_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    h_pool3, w_pool3 = math.ceil(h_pool2 / float(2)), math.ceil(w_pool2 / float(2))
    pool3, pool3_index, shape_3 = _max_pool_with_argmax(conv3_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool3', pad='SAME')
    # ------------------------End of encoder-----------------------------

    decoder_dropout3 = tf.layers.dropout(pool3, rate=(1 - dropout), name="decoder_dropout3")
    deconv3_1 = _up_pooling(decoder_dropout3, pool3_index, shape_3, h_pool3, w_pool3, batch_size, name="unpool_3")
    deconv3_2 = _conv_layer(deconv3_1, [3, 3, 256, 256], "deconv3_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv3_3 = _conv_layer(deconv3_2, [3, 3, 256, 256], "deconv3_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv3_4 = _conv_layer(deconv3_3, [3, 3, 256, 128], "deconv3_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    decoder_dropout2 = tf.layers.dropout(deconv3_4, rate=(1 - dropout), name="decoder_dropout2")
    deconv2_1 = _up_pooling(decoder_dropout2, pool2_index, shape_2, h_pool2, w_pool2, batch_size, name="unpool_2")
    deconv2_2 = _conv_layer(deconv2_1, [3, 3, 128, 128], "deconv2_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv2_3 = _conv_layer(deconv2_2, [3, 3, 128, 64], "deconv2_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    deconv1_1 = _up_pooling(deconv2_3, pool1_index, shape_1, h_pool1, w_pool1, batch_size, name="unpool_1")
    deconv1_2 = _conv_layer(deconv1_1, [3, 3, 64, 64], "deconv1_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv1_3 = _conv_layer(deconv1_2, [3, 3, 64, 64], "deconv1_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 64, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    if extract_features is True:
        return [tf.image.resize_bilinear(conv1_1, [32, 32]), 64], \
               [tf.image.resize_bilinear(conv3_2, [32, 32]), 256], \
               [tf.image.resize_bilinear(deconv1_3, [32, 32]), 64], conv_classifier
    else:
        return conv_classifier


def segnet_4(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features):
    x = tf.reshape(x, shape=[-1, crop, crop, num_input_bands])
    batch_size = tf.shape(x)[0]

    norm1 = tf.nn.local_response_normalization(x, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

    conv1_1 = _conv_layer(norm1, [3, 3, num_input_bands, 64], "conv1_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], "conv1_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    h_pool1, w_pool1 = crop_size, crop_size
    pool1, pool1_index, shape_1 = _max_pool_with_argmax(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool1', pad='SAME')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], "conv2_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], "conv2_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    h_pool2, w_pool2 = math.ceil(h_pool1 / float(2)), math.ceil(w_pool1 / float(2))
    pool2, pool2_index, shape_2 = _max_pool_with_argmax(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2', pad='SAME')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], "conv3_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], "conv3_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    h_pool3, w_pool3 = math.ceil(h_pool2 / float(2)), math.ceil(w_pool2 / float(2))
    pool3, pool3_index, shape_3 = _max_pool_with_argmax(conv3_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool3', pad='SAME')

    conv4_1 = _conv_layer(pool3, [3, 3, 128, 256], "conv4_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv4_2 = _conv_layer(conv4_1, [3, 3, 256, 256], "conv4_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    h_pool4, w_pool4 = math.ceil(h_pool3 / float(2)), math.ceil(w_pool3 / float(2))
    pool4, pool4_index, shape_4 = _max_pool_with_argmax(conv4_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool4', pad='SAME')

    # ------------------------End of encoder-----------------------------

    decoder_dropout4 = tf.layers.dropout(pool4, rate=(1 - dropout), name="decoder_dropout4")
    deconv4_1 = _up_pooling(decoder_dropout4, pool4_index, shape_4, h_pool4, w_pool4, batch_size, name="unpool_4")
    deconv4_2 = _conv_layer(deconv4_1, [3, 3, 256, 256], "deconv4_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv4_3 = _conv_layer(deconv4_2, [3, 3, 256, 256], "deconv4_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv4_4 = _conv_layer(deconv4_3, [3, 3, 256, 128], "deconv4_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    decoder_dropout3 = tf.layers.dropout(deconv4_4, rate=(1 - dropout), name="decoder_dropout3")
    deconv3_1 = _up_pooling(decoder_dropout3, pool3_index, shape_3, h_pool3, w_pool3, batch_size, name="unpool_3")
    deconv3_2 = _conv_layer(deconv3_1, [3, 3, 256, 256], "deconv3_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv3_3 = _conv_layer(deconv3_2, [3, 3, 256, 256], "deconv3_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv3_4 = _conv_layer(deconv3_3, [3, 3, 256, 128], "deconv3_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    decoder_dropout2 = tf.layers.dropout(deconv3_4, rate=(1 - dropout), name="decoder_dropout2")
    deconv2_1 = _up_pooling(decoder_dropout2, pool2_index, shape_2, h_pool2, w_pool2, batch_size, name="unpool_2")
    deconv2_2 = _conv_layer(deconv2_1, [3, 3, 128, 128], "deconv2_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv2_3 = _conv_layer(deconv2_2, [3, 3, 128, 64], "deconv2_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    deconv1_1 = _up_pooling(deconv2_3, pool1_index, shape_1, h_pool1, w_pool1, batch_size, name="unpool_1")
    deconv1_2 = _conv_layer(deconv1_1, [3, 3, 64, 64], "deconv1_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv1_3 = _conv_layer(deconv1_2, [3, 3, 64, 64], "deconv1_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 64, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


