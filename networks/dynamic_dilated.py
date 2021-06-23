from networks.layers import _conv_layer, _max_pool, _variable_with_weight_decay, _variable_on_cpu

import tensorflow as tf


def dilated_icpr_rate6(x, is_training, weight_decay, crop_size, num_input_bands, num_classes,
                       extract_features, batch_norm=True):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])

    conv1 = _conv_layer(x, [5, 5, num_input_bands, 64], "conv1", weight_decay, is_training,
                        rate=1, batch_norm=batch_norm, is_normal_conv=False)

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2,
                        batch_norm=batch_norm, is_normal_conv=False)

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], "conv3", weight_decay, is_training, rate=3,
                        batch_norm=batch_norm, is_normal_conv=False)

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4,
                        batch_norm=batch_norm, is_normal_conv=False)

    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5,
                        batch_norm=batch_norm, is_normal_conv=False)

    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6,
                        batch_norm=batch_norm, is_normal_conv=False)

    with tf.compat.v1.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_densely(x, is_training, weight_decay, crop_size, num_input_bands, num_classes,
                               extract_features, batch_norm=True):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])

    conv1 = _conv_layer(x, [5, 5, num_input_bands, 32], "conv1", weight_decay, is_training,
                        rate=1, batch_norm=batch_norm)

    conv2 = _conv_layer(conv1, [5, 5, 32, 32], 'conv2', weight_decay, is_training, rate=2,
                        batch_norm=batch_norm, is_normal_conv=False)
    try:
        c1 = tf.concat([conv1, conv2], 3)  # c1 = 32+32 = 64
    except:
        c1 = tf.concat(concat_dim=3, values=[conv1, conv2])

    conv3 = _conv_layer(c1, [4, 4, 64, 64], "conv3", weight_decay, is_training, rate=3,
                        batch_norm=batch_norm, is_normal_conv=False)
    try:
        c2 = tf.concat([c1, conv3], 3)  # c2 = 64+64 = 128
    except:
        c2 = tf.concat(concat_dim=3, values=[c1, conv3])

    conv4 = _conv_layer(c2, [4, 4, 128, 64], "conv4", weight_decay, is_training, rate=4,
                        batch_norm=batch_norm, is_normal_conv=False)
    try:
        c3 = tf.concat([c2, conv4], 3)  # c3 = 128+64 = 192
    except:
        c3 = tf.concat(concat_dim=3, values=[c2, conv4])

    conv5 = _conv_layer(c3, [3, 3, 192, 128], "conv5", weight_decay, is_training, rate=5,
                        batch_norm=batch_norm, is_normal_conv=False)
    try:
        c4 = tf.concat([c3, conv5], 3)  # c4 = 192+128 = 320
    except:
        c4 = tf.concat(concat_dim=3, values=[c3, conv5])

    conv6 = _conv_layer(c4, [3, 3, 320, 128], "conv6", weight_decay, is_training, rate=6,
                        batch_norm=batch_norm, is_normal_conv=False)
    try:
        c5 = tf.concat([c4, conv6], 3)  # c5 = 320+256 = 448
    except:
        c5 = tf.concat(concat_dim=3, values=[c4, conv6])

    with tf.compat.v1.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 448, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(c5, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_grsl(x, is_training, weight_decay, crop_size, num_input_bands, num_classes, extract_features):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])

    conv1 = _conv_layer(x, [5, 5, num_input_bands, 64], "conv1", weight_decay, is_training,
                        rate=1, activation='lrelu')
    pool1 = _max_pool(conv1, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool1')

    conv2 = _conv_layer(pool1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2,
                        activation='lrelu', is_normal_conv=False)
    pool2 = _max_pool(conv2, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool2')

    conv3 = _conv_layer(pool2, [4, 4, 64, 128], 'conv3', weight_decay, is_training, rate=3,
                        activation='lrelu', is_normal_conv=False)
    pool3 = _max_pool(conv3, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(pool3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4,
                        activation='lrelu', is_normal_conv=False)
    pool4 = _max_pool(conv4, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool4')

    conv5 = _conv_layer(pool4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5,
                        activation='lrelu', is_normal_conv=False)
    pool5 = _max_pool(conv5, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool5')

    conv6 = _conv_layer(pool5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6,
                        activation='lrelu', is_normal_conv=False)
    pool6 = _max_pool(conv6, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool6')

    with tf.compat.v1.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_grsl_rate8(x, is_training, weight_decay, crop_size, num_input_bands, num_classes, extract_features):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])

    conv1 = _conv_layer(x, [5, 5, num_input_bands, 64], "conv1", weight_decay, is_training,
                        rate=1, activation='lrelu')
    pool1 = _max_pool(conv1, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool1')

    conv2 = _conv_layer(pool1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2,
                        activation='lrelu', is_normal_conv=False)
    pool2 = _max_pool(conv2, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool2')

    conv3 = _conv_layer(pool2, [4, 4, 64, 128], 'conv3', weight_decay, is_training, rate=3,
                        activation='lrelu', is_normal_conv=False)
    pool3 = _max_pool(conv3, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(pool3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4,
                        activation='lrelu', is_normal_conv=False)
    pool4 = _max_pool(conv4, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool4')

    conv5 = _conv_layer(pool4, [3, 3, 128, 192], "conv5", weight_decay, is_training, rate=5,
                        activation='lrelu', is_normal_conv=False)
    pool5 = _max_pool(conv5, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool5')

    conv6 = _conv_layer(pool5, [3, 3, 192, 192], "conv6", weight_decay, is_training, rate=6,
                        activation='lrelu', is_normal_conv=False)
    pool6 = _max_pool(conv6, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool6')

    conv7 = _conv_layer(pool6, [3, 3, 192, 256], "conv7", weight_decay, is_training, rate=7,
                        activation='lrelu', is_normal_conv=False)
    pool7 = _max_pool(conv7, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool7')

    conv8 = _conv_layer(pool7, [3, 3, 256, 256], "conv8", weight_decay, is_training, rate=8,
                        activation='lrelu', is_normal_conv=False)
    pool8 = _max_pool(conv8, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool8')

    with tf.compat.v1.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, num_classes],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    if extract_features is True:
        return [tf.image.resize_bilinear(conv1, [32, 32]), 64], \
               [tf.image.resize_bilinear(pool5, [32, 32]), 192], \
               [tf.image.resize_bilinear(pool8, [32, 32]), 256], conv_classifier
    else:
        return conv_classifier
