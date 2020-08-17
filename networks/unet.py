import math

from networks.layers import _conv_layer, _deconv_layer, _max_pool, _crop_and_concat, \
    _variable_with_weight_decay, _variable_on_cpu

import tensorflow as tf


def _get_shape(_input):
    new_shape = [tf.shape(_input)[0], tf.shape(_input)[1]*2, tf.shape(_input)[2]*2, tf.shape(_input)[3]//2]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    return output_shape


# https://www.mdpi.com/2072-4292/11/9/1015/htm
def atrous_spatial_pyramid_pooling(inputs, atrous_rates, weight_decay, is_training):
    with tf.variable_scope("aspp"):
        # with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        #     with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]

        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = _conv_layer(inputs, [1, 1, 512, 256], "aspp_conv_1x1",
                               is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)
        conv_3x3_1 = _conv_layer(inputs, [3, 3, 512, 256], "aspp_conv_3x3_1", is_training=is_training,
                                 strides=[1, 1, 1, 1], rate=atrous_rates[0], is_normal_conv=False,
                                 weight_decay=weight_decay)
        conv_3x3_2 = _conv_layer(inputs, [3, 3, 512, 256], "aspp_conv_3x3_2", is_training=is_training,
                                 strides=[1, 1, 1, 1], rate=atrous_rates[1], is_normal_conv=False,
                                 weight_decay=weight_decay)
        conv_3x3_3 = _conv_layer(inputs, [3, 3, 512, 256], "aspp_conv_3x3_3", is_training=is_training,
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
            image_level_features = _conv_layer(image_level_features, [1, 1, 512, 256], "ilf_conv_1x1",
                                               is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)
            # bilinearly upsample features
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = _conv_layer(net, [1, 1, 256*5, 512], "conv_1x1_concat",
                          is_training=is_training, strides=[1, 1, 1, 1], weight_decay=weight_decay)

        return net


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


# https://www.mdpi.com/2072-4292/11/9/1015/htm
def unet_road_detection(x, dropout, is_training, weight_decay, crop, num_input_bands,
                        num_classes, crop_size, extract_features):
    x = tf.reshape(x, shape=[-1, crop, crop, num_input_bands])

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], "conv1_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], "conv1_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    s_pool1 = crop_size
    pool1 = _max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool1', pad='SAME')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], "conv2_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], "conv2_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    s_pool2 = math.ceil(s_pool1 / float(2))
    # s_pool2 = math.ceil(float(s_pool1 - 2 + 1) / float(2))
    pool2 = _max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2', pad='SAME')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], "conv3_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], "conv3_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    s_pool3 = math.ceil(s_pool2 / float(2))
    # s_pool3 = math.ceil(float(s_pool2 - 2 + 1) / float(2))
    pool3 = _max_pool(conv3_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool3', pad='SAME')

    conv4_1 = _conv_layer(pool3, [3, 3, 256, 512], "conv4_1", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    conv4_2 = _conv_layer(conv4_1, [3, 3, 512, 512], "conv4_2", weight_decay, is_training,
                          strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    # s_pool4 = math.ceil(s_pool3 / float(2))
    # pool4 = _max_pool(conv4_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool4', pad='SAME')
    #
    # conv5_1 = _conv_layer(pool4, [3, 3, 512, 1024], "conv5_1", weight_decay, is_training,
    #                       strides=[1, 1, 1, 1], pad='SAME', activation='elu')
    # conv5_2 = _conv_layer(conv5_1, [3, 3, 1024, 1024], "conv5_2", weight_decay, is_training,
    #                       strides=[1, 1, 1, 1], pad='SAME', activation='elu')

    # ---------------------------------End of encoder----------------------------------
    aspp = atrous_spatial_pyramid_pooling(conv4_2, [6, 12, 18], weight_decay, is_training)

    # deconv4_1 = _deconv_layer(aspp, [2, 2, 512, 1024], _get_shape(conv5_2), 'deconv4_1', weight_decay,
    #                           [1, 2, 2, 1], pad='SAME', has_bias=True)
    # # deconv4_2 = _crop_and_concat(conv2_2, deconv2_1, (s_pool2, s_pool2), (s_pool3*2, s_pool3*2))
    # deconv4_2 = tf.concat(values=[conv4_2, deconv4_1], axis=-1)
    # deconv4_3 = _conv_layer(deconv4_2, [3, 3, 1024, 512], "deconv4_3", weight_decay, is_training,
    #                         strides=[1, 1, 1, 1], pad='SAME')
    # deconv4_4 = _conv_layer(deconv4_3, [3, 3, 512, 512], "deconv4_4", weight_decay, is_training,
    #                         strides=[1, 1, 1, 1], pad='SAME')

    deconv3_1 = _deconv_layer(aspp, [2, 2, 256, 512], _get_shape(conv4_2), 'deconv3_1', weight_decay,
                              [1, 2, 2, 1], pad='SAME', has_bias=True)
    # deconv3_2 = _crop_and_concat(conv2_2, deconv2_1, (s_pool2, s_pool2), (s_pool3*2, s_pool3*2))
    deconv3_2 = tf.concat(values=[conv3_2, deconv3_1], axis=-1)
    deconv3_3 = _conv_layer(deconv3_2, [3, 3, 512, 256], "deconv3_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv3_4 = _conv_layer(deconv3_3, [3, 3, 256, 256], "deconv3_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    deconv2_1 = _deconv_layer(deconv3_4, [2, 2, 128, 256], _get_shape(conv3_2), 'deconv2_1', weight_decay,
                              [1, 2, 2, 1], pad='SAME', has_bias=True)
    # deconv2_2 = _crop_and_concat(conv2_2, deconv2_1, (s_pool2, s_pool2), (s_pool3*2, s_pool3*2))
    deconv2_2 = tf.concat(values=[conv2_2, deconv2_1], axis=-1)
    deconv2_3 = _conv_layer(deconv2_2, [3, 3, 256, 128], "deconv2_3", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')
    deconv2_4 = _conv_layer(deconv2_3, [3, 3, 128, 128], "deconv2_4", weight_decay, is_training,
                            strides=[1, 1, 1, 1], pad='SAME')

    deconv1_1 = _deconv_layer(deconv2_4, [2, 2, 64, 128], _get_shape(deconv2_4), 'deconv1_1', weight_decay,
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