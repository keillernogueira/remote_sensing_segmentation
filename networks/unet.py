import math

from networks.layers import _conv_layer, _deconv_layer, _max_pool, _crop_and_concat, \
    _variable_with_weight_decay, _variable_on_cpu

import tensorflow as tf


# https://github.com/jakeret/tf_unet/blob/master/tf_unet/unet.py
# https://github.com/zhulf0804/UNet-Tensorflow/blob/master/networks/unet.py
def unet(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features):
    x = tf.reshape(x, shape=[-1, crop, crop, num_input_bands])

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], "conv1_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], "conv1_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    s_pool1 = crop_size
    pool1 = _max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool1', pad='SAME')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], "conv2_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], "conv2_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    s_pool2 = math.ceil(s_pool1 / float(2))
    # s_pool2 = math.ceil(float(s_pool1 - 2 + 1) / float(2))
    pool2 = _max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2', pad='SAME')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], "conv3_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], "conv3_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    s_pool3 = math.ceil(s_pool2 / float(2))
    # s_pool3 = math.ceil(float(s_pool2 - 2 + 1) / float(2))
    print(s_pool1, s_pool2, s_pool3)

    # pool3 = _max_pool(conv3_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool3', pad='SAME')
    #
    # conv4_1 = _conv_layer(pool2, [3, 3, 128, 256], "conv3_1", weight_decay, is_training,
    #                       strides=[1, 1, 1, 1], pad='SAME')
    # conv4_2 = _conv_layer(conv3_1, [3, 3, 256, 256], "conv3_2", weight_decay, is_training,
    #                       strides=[1, 1, 1, 1], pad='SAME')

    # ------------------------End of encoder-----------------------------

    new_shape = [tf.shape(conv3_2)[0], tf.shape(conv3_2)[1]*2, tf.shape(conv3_2)[2]*2, tf.shape(conv3_2)[3]//2]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    deconv2_1 = _deconv_layer(conv3_2, [2, 2, 128, 256], output_shape, 'deconv2_1', weight_decay,
                              [1, 2, 2, 1], pad='SAME', has_bias=True)
    # deconv2_2 = _crop_and_concat(conv2_2, deconv2_1, (s_pool2, s_pool2), (s_pool3*2, s_pool3*2))
    deconv2_2 = tf.concat(values=[conv2_2, deconv2_1], axis=-1)
    deconv2_3 = _conv_layer(deconv2_2, [3, 3, 256, 128], "deconv2_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv2_4 = _conv_layer(deconv2_3, [3, 3, 128, 128], "deconv2_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    new_shape = [tf.shape(deconv2_4)[0], tf.shape(deconv2_4)[1]*2, tf.shape(deconv2_4)[2]*2, tf.shape(deconv2_4)[3]//2]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    deconv1_1 = _deconv_layer(deconv2_4, [2, 2, 64, 128], output_shape, 'deconv1_1', weight_decay,
                              [1, 2, 2, 1], pad='SAME', has_bias=True)
    # deconv1_2 = _crop_and_concat(conv1_2, deconv1_1, (s_pool2, s_pool2), (s_pool2, s_pool2))
    deconv1_2 = tf.concat(values=[conv1_2, deconv1_1], axis=-1)
    deconv1_3 = _conv_layer(deconv1_2, [3, 3, 128, 64], "deconv1_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv1_4 = _conv_layer(deconv1_3, [3, 3, 64, 64], "deconv1_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 64, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(deconv1_4, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def unet_4(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features):
    x = tf.reshape(x, shape=[-1, crop, crop, num_input_bands])

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], "conv1_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], "conv1_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    s_pool1 = crop_size
    pool1 = _max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool1', pad='SAME')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], "conv2_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], "conv2_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    s_pool2 = math.ceil(s_pool1 / float(2))
    # s_pool2 = math.ceil(float(s_pool1 - 2 + 1) / float(2))
    pool2 = _max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2', pad='SAME')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], "conv3_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], "conv3_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    s_pool3 = math.ceil(s_pool2 / float(2))
    # s_pool3 = math.ceil(float(s_pool2 - 2 + 1) / float(2))
    pool3 = _max_pool(conv3_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool3', pad='SAME')

    conv4_1 = _conv_layer(pool3, [3, 3, 256, 256], "conv4_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv4_2 = _conv_layer(conv4_1, [3, 3, 256, 256], "conv4_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    # s_pool4 = math.ceil(s_pool3 / float(2))

    # ------------------------End of encoder-----------------------------

    new_shape = [tf.shape(conv4_2)[0], tf.shape(conv4_2)[1]*2, tf.shape(conv4_2)[2]*2, tf.shape(conv4_2)[3]//2]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    deconv3_1 = _deconv_layer(conv4_2, [2, 2, 128, 256], output_shape, 'deconv3_1', weight_decay,
                              [1, 2, 2, 1], pad='SAME', has_bias=True)
    # deconv3_2 = _crop_and_concat(conv2_2, deconv2_1, (s_pool2, s_pool2), (s_pool3*2, s_pool3*2))
    deconv3_2 = tf.concat(values=[conv3_2, deconv3_1], axis=-1)
    deconv3_3 = _conv_layer(deconv3_2, [3, 3, 384, 256], "deconv3_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv3_4 = _conv_layer(deconv3_3, [3, 3, 256, 256], "deconv3_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    new_shape = [tf.shape(conv3_2)[0], tf.shape(conv3_2)[1]*2, tf.shape(conv3_2)[2]*2, tf.shape(conv3_2)[3]//2]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    deconv2_1 = _deconv_layer(deconv3_4, [2, 2, 128, 256], output_shape, 'deconv2_1', weight_decay,
                              [1, 2, 2, 1], pad='SAME', has_bias=True)
    # deconv2_2 = _crop_and_concat(conv2_2, deconv2_1, (s_pool2, s_pool2), (s_pool3*2, s_pool3*2))
    deconv2_2 = tf.concat(values=[conv2_2, deconv2_1], axis=-1)
    deconv2_3 = _conv_layer(deconv2_2, [3, 3, 256, 128], "deconv2_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv2_4 = _conv_layer(deconv2_3, [3, 3, 128, 128], "deconv2_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    new_shape = [tf.shape(deconv2_4)[0], tf.shape(deconv2_4)[1]*2, tf.shape(deconv2_4)[2]*2, tf.shape(deconv2_4)[3]//2]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    deconv1_1 = _deconv_layer(deconv2_4, [2, 2, 64, 128], output_shape, 'deconv1_1', weight_decay,
                              [1, 2, 2, 1], pad='SAME', has_bias=True)
    # deconv1_2 = _crop_and_concat(conv1_2, deconv1_1, (s_pool2, s_pool2), (s_pool2, s_pool2))
    deconv1_2 = tf.concat(values=[conv1_2, deconv1_1], axis=-1)
    deconv1_3 = _conv_layer(deconv1_2, [3, 3, 128, 64], "deconv1_2", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv1_4 = _conv_layer(deconv1_3, [3, 3, 64, 64], "deconv1_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 64, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(deconv1_4, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier
