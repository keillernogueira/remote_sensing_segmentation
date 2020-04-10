from networks.layers import _conv_layer, _max_pool

import tensorflow as tf


# parameters: 2.228.224
def atrous_spatial_pyramid_pooling(inputs, atrous_rates, weight_decay, is_training):
    with tf.variable_scope("aspp"):
        # with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        #     with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]

        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = _conv_layer(inputs, [1, 1, 256, 256], "aspp_conv_1x1", is_training=is_training, strides=[1, 1, 1, 1],
                               weight_decay=weight_decay)
        conv_3x3_1 = _conv_layer(inputs, [3, 3, 256, 256], "aspp_conv_3x3_1", is_training=is_training,
                                 strides=[1, 1, 1, 1], rate=atrous_rates[0], is_normal_conv=False,
                                 weight_decay=weight_decay)
        conv_3x3_2 = _conv_layer(inputs, [3, 3, 256, 256], "aspp_conv_3x3_2", is_training=is_training,
                                 strides=[1, 1, 1, 1], rate=atrous_rates[1], is_normal_conv=False,
                                 weight_decay=weight_decay)
        conv_3x3_3 = _conv_layer(inputs, [3, 3, 256, 256], "aspp_conv_3x3_3", is_training=is_training,
                                 strides=[1, 1, 1, 1], rate=atrous_rates[2], is_normal_conv=False,
                                 weight_decay=weight_decay)

        # conv_3x3_1 = _conv_layer(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0],
        #                                scope='conv_3x3_1')
        # conv_3x3_2 = _conv_layer(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1],
        #                                scope='conv_3x3_2')
        # conv_3x3_3 = _conv_layer(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2],
        #                                scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
            # global average pooling
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
            # 1x1 convolution with 256 filters( and batch normalization)
            image_level_features = _conv_layer(image_level_features, [1, 1, 256, 256], "ilf_conv_1x1",
                                               is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)
            # bilinearly upsample features
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = _conv_layer(net, [1, 1, 256*5, 256], "conv_1x1_concat",
                          is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)

        return net


def deeplab(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size):
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], 'conv1_1', weight_decay, is_training, batch_norm=True)
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], 'conv1_2', weight_decay, is_training, batch_norm=True)
    pool1 = _max_pool(conv1_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_1')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], 'conv2_1', weight_decay, is_training, batch_norm=True)
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], 'conv2_2', weight_decay, is_training, batch_norm=True)
    pool2 = _max_pool(conv2_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_2')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], 'conv3_1', weight_decay, is_training, batch_norm=True)
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], 'conv3_2', weight_decay, is_training, batch_norm=True)
    pool3 = _max_pool(conv3_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_3')

    encoder_output = atrous_spatial_pyramid_pooling(pool3, [2, 2, 2], weight_decay, is_training)

    with tf.variable_scope("decoder"):
        # with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        #     with arg_scope([layers.batch_norm], is_training=is_training):
        with tf.variable_scope("low_level_features"):
            low_level_features = pool1
            low_level_features = _conv_layer(low_level_features, [1, 1, 64, 256], "conv_1x1",
                                             is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)
            # conv2d(low_level_features, 48, [1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]

        with tf.variable_scope("upsampling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = _conv_layer(net, [3, 3, 256*2, 256], "conv_3x3_1",
                              is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)
            net = _conv_layer(net, [3, 3, 256, 256], "conv_3x3_2",
                              is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)
            net = _conv_layer(net, [1, 1, 256, num_classes], "conv_1x1", is_training=is_training, batch_norm=False,
                              has_activation=False, strides=[1, 1, 1, 1], weight_decay=weight_decay)

            # net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
            # net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
            # net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, tf.shape(x)[1:3], name='upsample_2')

    return logits
