import math
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


def _add_wd(var, wd, collection_name='losses'):
    if not tf.get_variable_scope().reuse:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return var


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


def _fc_layer(input, layer_shape, weight_decay, name, batch_norm=False, is_training=None, activation=None):
    with tf.compat.v1.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=layer_shape,
                                              ini=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', layer_shape[-1], tf.constant_initializer(0.1))

        fc = tf.matmul(input, weights)
        fc = tf.add(fc, biases)

        if batch_norm is True:
            fc = _batch_norm(fc, is_training, scope=scope)

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
                activation='relu', batch_norm=True, has_activation=True, is_normal_conv=True,
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
            elif activation == 'elu':
                conv_act = tf.nn.elu(conv_act, name=name)
            else:
                conv_act = leaky_relu(conv_act)

        return conv_act


def get_deconv_filter(final_shape):
    width = final_shape[0]
    heigh = final_shape[0]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([final_shape[0], final_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(final_shape)
    for i in range(final_shape[2]):
        for j in range(final_shape[3]):
            weights[:, :, i, j] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)


def _deconv_layer(input, layer_shape, output_shape, name, weight_decay, strides=None,
                  pad='SAME', has_bias=False, debug=False):
    '''
        #### Estimate output of deconv
        import tensorflow as tf
        sess = tf.Session()
        output_shape = [1,25,25,2]
        strides = [1, 4, 4, 1]

        l = tf.constant(0.1, shape=[1, 7, 7, 1024])
        w = tf.constant(0.1, shape=[4, 4, 2, 1024])

        h1 = tf.nn.conv2d_transpose(l, w, output_shape=output_shape, strides=strides, padding='SAME')
        print sess.run(h1)

        output = tf.constant(0.1, shape=output_shape)
        expected_l = tf.nn.conv2d(output, w, strides=strides, padding='SAME')
        print expected_l.get_shape()
    '''
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name):
        # logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        # final_shape = [ksize, ksize, NUM_CLASSES, in_features]

        weights = get_deconv_filter(layer_shape)
        _add_wd(weights, weight_decay)
        deconv = tf.nn.conv2d_transpose(input, weights, output_shape, strides=strides, padding=pad)

        if has_bias is True:
            biases = _variable_on_cpu('biases', layer_shape[-2], tf.constant_initializer(0.1))
            deconv = tf.nn.bias_add(deconv, biases)

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)], message='Shape of %s' % name, summarize=4, first_n=1)

    return deconv


def _up_pooling(pool, ind, output_shape, h, w, batch_size, name=None):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
           :param batch_size:
    """
    with tf.variable_scope(name):
        pool_ = tf.reshape(pool, [-1])
        batch_range = tf.reshape(tf.range(tf.cast(batch_size, dtype=ind.dtype), dtype=ind.dtype),
                                 [tf.shape(pool)[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [-1, 1])
        ind_ = tf.reshape(ind, [-1, 1])
        ind_ = tf.concat([b, ind_], 1)

        # pool_ = tf.Print(pool_, [tf.shape(pool_)], message='Shape of pool_')
        # ind_ = tf.Print(ind_, [tf.shape(ind_)], message='Shape of ind_')

        ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, h * w * output_shape[3]])
        # the reason that we use tf.scatter_nd: if we use tf.sparse_tensor_to_dense,
        # then the gradient is None, which will cut off the network.
        # But if we use tf.scatter_nd, the gradients for all the trainable variables will be tensors, instead of None.
        # The usage for tf.scatter_nd is that: create a new tensor by applying sparse
        # UPDATES(which is the pooling value) to individual values of slices within a
        # zero tensor of given shape (FLAT_OUTPUT_SHAPE) according to the indices (ind_).
        # If we ues the orignal code, the only thing we need to change is: changeing
        # from tf.sparse_tensor_to_dense(sparse_tensor) to tf.sparse_add(tf.zeros((output_sahpe)),
        # sparse_tensor) which will give us the gradients!!!
        ret = tf.reshape(ret, [tf.shape(pool)[0], h, w, output_shape[3]])
        return ret


def _crop_and_concat(x1, x2, x1_shape, x2_shape):
    with tf.name_scope("crop_and_concat"):
        # x1_shape = tf.shape(x1)
        # x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def _max_pool(input_data, kernel, strides, name, pad='SAME', debug=False):
    pool = tf.nn.max_pool2d(input_data, ksize=kernel, strides=strides, padding=pad, name=name)
    if debug:
        pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

    return pool


def _max_pool_with_argmax(input, kernel, strides, name, pad='SAME'):
    with tf.variable_scope(name) as scope:
        value, index = tf.nn.max_pool_with_argmax(tf.to_double(input), ksize=kernel, strides=strides,
                                                  padding=pad, name=scope.name)
    return tf.to_float(value), index, input.get_shape().as_list()


def _avg_pool(input_data, kernel, strides, name, pad='SAME', debug=False):
    pool = tf.nn.avg_pool(input_data, ksize=kernel, strides=strides, padding=pad, name=name)
    if debug:
        pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

    return pool
