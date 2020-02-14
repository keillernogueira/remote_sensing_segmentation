from networks.layers import _conv_layer, _deconv_layer, _max_pool, _crop_and_concat, \
    _variable_with_weight_decay, _variable_on_cpu

import tensorflow as tf


# https://github.com/jakeret/tf_unet/blob/master/tf_unet/unet.py
# https://github.com/zhulf0804/UNet-Tensorflow/blob/master/networks/unet.py
def unet_25(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes):
    x = tf.reshape(x, shape=[-1, crop, crop, num_input_bands])

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], "conv1_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], "conv1_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    pool1 = _max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool1', pad='SAME')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], "conv2_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], "conv2_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    pool2 = _max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2', pad='SAME')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], "conv3_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], "conv3_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME')
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
    deconv2_2 = _crop_and_concat(conv2_2, deconv2_1)
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
    deconv1_2 = _crop_and_concat(conv1_2, deconv1_1)
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
