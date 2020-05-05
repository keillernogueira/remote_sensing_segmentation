from networks.layers import _conv_layer, _max_pool, _deconv_layer

import tensorflow as tf


def fcn_25_1_4x(x, dropout, is_training, crop_size, weight_decay, num_input_bands, num_classes):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], 'conv1_1', weight_decay, is_training, batch_norm=True)
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], 'conv1_2', weight_decay, is_training, batch_norm=True)
    pool1 = _max_pool(conv1_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_1')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], 'conv2_1', weight_decay, is_training, batch_norm=True)
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], 'conv2_2', weight_decay, is_training, batch_norm=True)
    pool2 = _max_pool(conv2_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_2')

    # reshape = tf.reshape(pool3, [-1, 7*7*256])
    # fc1 = _fc_layer(pool2, layerShape=[3, 3, 128, 1024], name='fc1', weight_decay=weight_decay)
    fc1 = _conv_layer(pool2, layer_shape=[3, 3, 128, 1024], name='fc1', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = _fc_layer(drop_fc1, layerShape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay)
    fc2 = _conv_layer(drop_fc1, layer_shape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc2 = tf.nn.dropout(fc2, dropout)

    # score_fr = _score_layer(drop_fc2, 'score_fr', weight_decay)
    score_fr = _conv_layer(drop_fc2, [1, 1, drop_fc2.get_shape()[3].value, num_classes],
                           'score_fr', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME', batch_norm=False,
                           has_activation=False)

    # final_shape = [ksize, ksize, NUM_CLASSES, in_features]
    new_shape = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore3 = _upscore_layer(score_fr, layerShape=tf.shape(x), name='upscore3', weight_decay=weight_decay, ksize=8,
    #                           stride=4)
    upscore3 = _deconv_layer(score_fr, [8, 8, num_classes, score_fr.get_shape()[3].value], output_shape,
                             'upscore3', weight_decay, strides=[1, 4, 4, 1], pad='SAME')

    return upscore3


def fcn_25_2_2x(x, dropout, is_training, crop_size, weight_decay, num_input_bands, num_classes):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])
    # print x.get_shape()

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], 'conv1_1', weight_decay, is_training, batch_norm=True)
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], 'conv1_2', weight_decay, is_training, batch_norm=True)
    pool1 = _max_pool(conv1_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_1')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], 'conv2_1', weight_decay, is_training, batch_norm=True)
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], 'conv2_2', weight_decay, is_training, batch_norm=True)
    pool2 = _max_pool(conv2_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_2')

    # reshape = tf.reshape(pool3, [-1, 7*7*256])
    # fc1 = _fc_layer(pool2, layerShape=[3, 3, 128, 1024], name='fc1', weight_decay=weight_decay)
    fc1 = _conv_layer(pool2, layer_shape=[3, 3, 128, 1024], name='fc1', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = _fc_layer(drop_fc1, layerShape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay)
    fc2 = _conv_layer(drop_fc1, layer_shape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc2 = tf.nn.dropout(fc2, dropout)

    # score_fr = _score_layer(drop_fc2, 'score_fr', weight_decay)
    score_fr = _conv_layer(drop_fc2, [1, 1, drop_fc2.get_shape()[3].value, num_classes],
                           'score_fr', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME', batch_norm=False,
                           has_activation=False)

    new_shape = [tf.shape(pool1)[0], tf.shape(pool1)[1], tf.shape(pool1)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore2 = _upscore_layer(score_fr, layerShape=tf.shape(pool1), name='upscore2',
    # weight_decay=weight_decay, ksize=4, stride=2)
    upscore2 = _deconv_layer(score_fr, [4, 4, num_classes, score_fr.get_shape()[3].value], output_shape,
                             'upscore2', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    # score_pool4 = _score_layer(pool1, "score_pool1", weight_decay)
    score_pool4 = _conv_layer(pool1, [1, 1, pool1.get_shape()[3].value, num_classes],
                              'score_pool1', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                              batch_norm=False, has_activation=False)
    fuse_pool4 = tf.add(upscore2, score_pool4)

    new_shape = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore3 = _upscore_layer(fuse_pool4, layerShape=tf.shape(x), name='upscore3', weight_decay=weight_decay,
    #                           ksize=4, stride=2)
    upscore3 = _deconv_layer(fuse_pool4, [4, 4, num_classes, fuse_pool4.get_shape()[3].value], output_shape,
                             'upscore3', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    return upscore3


def fcn_25_3_2x_icpr(x, dropout, is_training, crop_size, weight_decay, num_input_bands, num_classes):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])
    # print x.get_shape()

    conv1 = _conv_layer(x, [4, 4, num_input_bands, 64], 'conv1', weight_decay, is_training, batch_norm=True)
    pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')

    conv2 = _conv_layer(pool1, [4, 4, 64, 128], 'conv2', weight_decay, is_training, batch_norm=True)
    pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

    conv3 = _conv_layer(pool2, [3, 3, 128, 256], 'conv3', weight_decay, is_training, batch_norm=True)
    pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='pool3')

    # fc1 = _fc_layer(pool3, layerShape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay)
    fc1 = _conv_layer(pool3, layer_shape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = _fc_layer(drop_fc1, layerShape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay)
    fc2 = _conv_layer(drop_fc1, layer_shape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc2 = tf.nn.dropout(fc2, dropout)

    # score_fr = _score_layer(drop_fc2, 'score_fr', weight_decay)
    score_fr = _conv_layer(drop_fc2, [1, 1, drop_fc2.get_shape()[3].value, num_classes],
                           'score_fr', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                           batch_norm=False, has_activation=False)

    new_shape = [tf.shape(pool2)[0], tf.shape(pool2)[1], tf.shape(pool2)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore4 = _upscore_layer(score_fr, layerShape=tf.shape(pool2), name='upscore4', weight_decay=weight_decay,
    #                           ksize=4, stride=1)
    upscore4 = _deconv_layer(score_fr, [4, 4, num_classes, score_fr.get_shape()[3].value], output_shape,
                             'upscore4', weight_decay, strides=[1, 1, 1, 1], pad='SAME')

    # score_pool2 = _score_layer(pool2, "score_pool2", weight_decay)
    score_pool2 = _conv_layer(pool2, [1, 1, pool2.get_shape()[3].value, num_classes],
                              'score_pool2', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                              batch_norm=False, has_activation=False)
    fuse_pool2 = tf.add(upscore4, score_pool2)

    new_shape = [tf.shape(pool1)[0], tf.shape(pool1)[1], tf.shape(pool1)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore2 = _upscore_layer(fuse_pool2, layerShape=tf.shape(pool1), name='upscore2', weight_decay=weight_decay,
    #                           ksize=4, stride=2)
    upscore2 = _deconv_layer(fuse_pool2, [4, 4, num_classes, fuse_pool2.get_shape()[3].value], output_shape,
                             'upscore2', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    # score_pool1 = _score_layer(pool1, "score_pool1", weight_decay)
    score_pool1 = _conv_layer(pool1, [1, 1, pool1.get_shape()[3].value, num_classes],
                              'score_pool1', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                              batch_norm=False, has_activation=False)
    fuse_pool1 = tf.add(upscore2, score_pool1)

    new_shape = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore = _upscore_layer(fuse_pool1, layerShape=tf.shape(x), name='upscore7', weight_decay=weight_decay,
    #                          ksize=4, stride=2)
    upscore = _deconv_layer(fuse_pool1, [4, 4, num_classes, fuse_pool1.get_shape()[3].value], output_shape,
                            'upscore7', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    return upscore


def fcn_50_1_8x(x, dropout, is_training, crop_size, weight_decay, num_input_bands, num_classes):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, num_input_bands])
    # print x.get_shape()

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], 'conv1_1', weight_decay, is_training, batch_norm=True)
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], 'conv1_2', weight_decay, is_training, batch_norm=True)
    pool1 = _max_pool(conv1_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_1')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], 'conv2_1', weight_decay, is_training, batch_norm=True)
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], 'conv2_2', weight_decay, is_training, batch_norm=True)
    pool2 = _max_pool(conv2_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_2')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], 'conv3_1', weight_decay, is_training, batch_norm=True)
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], 'conv3_2', weight_decay, is_training, batch_norm=True)
    pool3 = _max_pool(conv3_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_3')

    # reshape = tf.reshape(pool3, [-1, 7*7*256])
    # fc1 = _fc_layer(pool3, layerShape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay)
    fc1 = _conv_layer(pool3, layer_shape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = _fc_layer(drop_fc1, layerShape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay)
    fc2 = _conv_layer(drop_fc1, layer_shape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc2 = tf.nn.dropout(fc2, dropout)

    # score_fr = _score_layer(drop_fc2, 'score_fr', weight_decay)
    score_fr = _conv_layer(drop_fc2, [1, 1, drop_fc2.get_shape()[3].value, num_classes],
                           'score_fr', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME', batch_norm=False,
                           has_activation=False)

    new_shape = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore = _upscore_layer(score_fr, layerShape=tf.shape(x), name='upscore7', weight_decay=weight_decay,
    #                          ksize=16, stride=8)
    upscore = _deconv_layer(score_fr, [16, 16, num_classes, score_fr.get_shape()[3].value], output_shape,
                            'upscore7', weight_decay, strides=[1, 8, 8, 1], pad='SAME')

    return upscore


def fcn_50_2_4x(x, dropout, is_training, crop_size, weight_decay, num_input_bands, num_classes):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, is_training, crop_size, crop_size, num_input_bands])
    # print x.get_shape()

    conv1_1 = _conv_layer(x, [3, 3, num_input_bands, 64], 'conv1_1', weight_decay, is_training, batch_norm=True)
    conv1_2 = _conv_layer(conv1_1, [3, 3, 64, 64], 'conv1_2', weight_decay, is_training, batch_norm=True)
    pool1 = _max_pool(conv1_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_1')

    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], 'conv2_1', weight_decay, is_training, batch_norm=True)
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], 'conv2_2', weight_decay, is_training, batch_norm=True)
    pool2 = _max_pool(conv2_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_2')

    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], 'conv3_1', weight_decay, is_training, batch_norm=True)
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], 'conv3_2', weight_decay, is_training, batch_norm=True)
    pool3 = _max_pool(conv3_2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_3')

    # reshape = tf.reshape(pool3, [-1, 7*7*256])
    # fc1 = _fc_layer(pool3, layerShape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay)
    fc1 = _conv_layer(pool3, layer_shape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = _fc_layer(drop_fc1, layerShape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay)
    fc2 = _conv_layer(drop_fc1, layer_shape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc2 = tf.nn.dropout(fc2, dropout)

    # score_fr = _score_layer(drop_fc2, 'score_fr', weight_decay)
    score_fr = _conv_layer(drop_fc2, [1, 1, drop_fc2.get_shape()[3].value, num_classes],
                           'score_fr', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME', batch_norm=False,
                           has_activation=False)

    new_shape = [tf.shape(pool2)[0], tf.shape(pool2)[1], tf.shape(pool2)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore4 = _upscore_layer(score_fr, layerShape=tf.shape(pool2), name='upscore4', weight_decay=weight_decay,
    #                           ksize=4, stride=2)
    upscore4 = _deconv_layer(score_fr, [4, 4, num_classes, score_fr.get_shape()[3].value], output_shape,
                             'upscore4', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    # score_pool2 = _score_layer(pool2, "score_pool2", weight_decay)
    score_pool2 = _conv_layer(pool2, [1, 1, pool2.get_shape()[3].value, num_classes],
                              'score_pool2', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                              batch_norm=False, has_activation=False)
    fuse_pool2 = tf.add(upscore4, score_pool2)

    new_shape = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore = _upscore_layer(fuse_pool2, layerShape=tf.shape(x), name='upscore7', weight_decay=weight_decay,
    #                          ksize=8, stride=4)
    upscore = _deconv_layer(fuse_pool2, [8, 8, num_classes, fuse_pool2.get_shape()[3].value], output_shape,
                            'upscore7', weight_decay, strides=[1, 4, 4, 1], pad='SAME')

    return upscore


def fcn_50_3_2x(x, dropout, is_training, crop_size, weight_decay, num_input_bands, num_classes):
    # Reshape input picture
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

    # reshape = tf.reshape(pool3, [-1, 7*7*256])
    # fc1 = _fc_layer(pool3, layerShape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay)
    fc1 = _conv_layer(pool3, layer_shape=[3, 3, 256, 1024], name='fc1', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc1 = tf.nn.dropout(fc1, dropout)

    # fc2 = _fc_layer(drop_fc1, layerShape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay)
    fc2 = _conv_layer(drop_fc1, layer_shape=[1, 1, 1024, 1024], name='fc2', weight_decay=weight_decay,
                      is_training=is_training, strides=[1, 1, 1, 1], pad='SAME', activation='relu',
                      batch_norm=False, has_activation=True)
    drop_fc2 = tf.nn.dropout(fc2, dropout)

    # score_fr = _score_layer(drop_fc2, 'score_fr', weight_decay)
    score_fr = _conv_layer(drop_fc2, [1, 1, drop_fc2.get_shape()[3].value, num_classes],
                           'score_fr', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                           batch_norm=False, has_activation=False)

    new_shape = [tf.shape(pool2)[0], tf.shape(pool2)[1], tf.shape(pool2)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore4 = _upscore_layer(score_fr, layerShape=tf.shape(pool2), name='upscore4', weight_decay=weight_decay,
    #                           ksize=4, stride=2)
    upscore4 = _deconv_layer(score_fr, [4, 4, num_classes, score_fr.get_shape()[3].value], output_shape,
                             'upscore4', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    # score_pool2 = _score_layer(pool2, "score_pool2", weight_decay)
    score_pool2 = _conv_layer(pool2, [1, 1, pool2.get_shape()[3].value, num_classes],
                              'score_pool2', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                              batch_norm=False, has_activation=False)
    fuse_pool2 = tf.add(upscore4, score_pool2)

    new_shape = [tf.shape(pool1)[0], tf.shape(pool1)[1], tf.shape(pool1)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore2 = _upscore_layer(fuse_pool2, layerShape=tf.shape(pool1), name='upscore2', weight_decay=weight_decay,
    #                           ksize=4, stride=2)
    upscore2 = _deconv_layer(fuse_pool2, [4, 4, num_classes, fuse_pool2.get_shape()[3].value], output_shape,
                             'upscore2', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    # score_pool1 = _score_layer(pool1, "score_pool1", weight_decay)
    score_pool1 = _conv_layer(pool1, [1, 1, pool1.get_shape()[3].value, num_classes],
                              'score_pool1', weight_decay, is_training, strides=[1, 1, 1, 1], pad='SAME',
                              batch_norm=False, has_activation=False)
    fuse_pool1 = tf.add(upscore2, score_pool1)

    new_shape = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], num_classes]
    try:
        output_shape = tf.pack(new_shape)
    except:
        output_shape = tf.stack(new_shape)
    # upscore = _upscore_layer(fuse_pool1, layerShape=tf.shape(x), name='upscore7', weight_decay=weight_decay,
    #                          ksize=4, stride=2)
    upscore = _deconv_layer(fuse_pool1, [4, 4, num_classes, fuse_pool1.get_shape()[3].value], output_shape,
                            'upscore', weight_decay, strides=[1, 2, 2, 1], pad='SAME')

    return upscore
