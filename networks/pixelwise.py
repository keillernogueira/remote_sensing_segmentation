from networks.layers import _conv_layer, _max_pool, _fc_layer, _variable_with_weight_decay, _variable_on_cpu

import tensorflow as tf


def pixelwise(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes):
    x = tf.reshape(x, shape=[-1, crop, crop, num_input_bands])

    conv1 = _conv_layer(x, [4, 4, num_input_bands, 64], "conv1", weight_decay, is_training, batch_norm=True)
    pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')

    conv2 = _conv_layer(pool1, [4, 4, 64, 128], "conv2", weight_decay, is_training, batch_norm=True)
    pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

    conv3 = _conv_layer(pool2, [3, 3, 128, 256], "conv3", weight_decay, is_training, batch_norm=True)
    pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='pool3')

    reshape = tf.reshape(pool3, [-1, 1 * 1 * 256])
    drop_fc1 = tf.nn.dropout(reshape, dropout)
    fc1 = _fc_layer(drop_fc1, [1 * 1 * 256, 1024], weight_decay, 'fc1', batch_norm=True,
                    is_training=is_training, activation='relu')

    drop_fc2 = tf.nn.dropout(fc1, dropout)
    fc2 = _fc_layer(drop_fc2, [1024, 1024], weight_decay, 'fc2', batch_norm=True,
                    is_training=is_training, activation='relu')

    # Output, class prediction
    with tf.variable_scope('fc3_logits') as scope:
        weights = _variable_with_weight_decay('weights', [1024, num_classes],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return logits

