import numpy as np

import tensorflow as tf


def leaky_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def identity_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Identity matrix initializer can only be used for 2D square matrices.')
        else:
            return tf.constant_op.constant(scale * np.identity(shape[0], dtype), dtype=dtype)

    return _initializer


def _variable_on_cpu(name, shape, ini):
    with tf.device('/cpu:0'):
        var = tf.compat.v1.get_variable(name, shape, initializer=ini, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, ini, weight_decay):
    var = _variable_on_cpu(name, shape, ini)
    # tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    # tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    # tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    # orthogonal_initializer()
    if weight_decay is not None:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_decay)
    return var


def _batch_norm(input_data, is_training, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=True, center=False,
                                                        updates_collections=None,
                                                        scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=False, center=False,
                                                        updates_collections=None, scope=scope, reuse=True)
                   )


def _fc_layer(input, layer_shape, weight_decay, name, activation=None):
    with tf.compat.v1.variable_scope(name):
        weights = _variable_with_weight_decay('weights', shape=layer_shape,
                                              ini=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', layer_shape[-1], tf.constant_initializer(0.1))

        fc = tf.matmul(input, weights)
        fc = tf.add(fc, biases)

        if activation == 'relu':
            fc = tf.nn.relu(fc, name=name)

        return fc


def _squeeze_excitation_layer(input_data, weight_decay, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = tf.reduce_mean(input_data, axis=[1,2])
        out_dim = squeeze.get_shape().as_list()[1]
        # print(squeeze.get_shape().as_list(), out_dim)

        excitation = _fc_layer(squeeze, [out_dim, out_dim/ratio], weight_decay, layer_name+'_fc1')
        excitation = tf.nn.relu(excitation)
        excitation = _fc_layer(excitation, [out_dim/ratio, out_dim], weight_decay, name=layer_name+'_fc2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = input_data * excitation

    return scale


def _conv_layer(input_data, layer_shape, name, weight_decay, is_training, rate=1, strides=None, pad='SAME',
                activation='relu', batch_norm=True, has_activation=True, is_normal_conv=False,
                init_func=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.compat.v1.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=layer_shape, ini=init_func, weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', layer_shape[-1], tf.constant_initializer(0.1))

        if is_normal_conv is False:
            conv_op = tf.nn.atrous_conv2d(input_data, weights, rate=rate, padding=pad)
        else:
            conv_op = tf.nn.conv2d(input_data, weights, strides=strides, padding=pad)
        conv_act = tf.nn.bias_add(conv_op, biases)

        if batch_norm is True:
            conv_act = _batch_norm(conv_act, is_training, scope=scope)
        if has_activation is True:
            if activation == 'relu':
                conv_act = tf.nn.relu(conv_act, name=name)
            else:
                conv_act = leaky_relu(conv_act)

        return conv_act


def _max_pool(input_data, kernel, strides, name, pad='SAME', debug=False):
    pool = tf.nn.max_pool2d(input_data, ksize=kernel, strides=strides, padding=pad, name=name)
    if debug:
        pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

    return pool


def _avg_pool(input_data, kernel, strides, name, pad='SAME', debug=False):
    pool = tf.nn.avg_pool(input_data, ksize=kernel, strides=strides, padding=pad, name=name)
    if debug:
        pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

    return pool
