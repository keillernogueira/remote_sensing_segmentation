import tensorflow as tf


NUM_CLASSES = 2


def loss_def(model, _logits, _labels, _mask=None, _ssim=False, _pred=None):
    if model != 'pixelwise':
        # logits = tf.reshape(_logits, [-1, NUM_CLASSES])
        # labels = tf.cast(tf.reshape(_labels, [-1]), tf.int32)

        mask = tf.cast(tf.reshape(_mask, [-1]), tf.bool)
        logits = tf.boolean_mask(tf.reshape(_logits, [-1, NUM_CLASSES]), mask)
        labels = tf.boolean_mask(tf.cast(tf.reshape(_labels, [-1]), tf.int64), mask)
    else:
        logits = _logits
        labels = _labels

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.compat.v1.add_to_collection('losses', cross_entropy_mean)

    if _ssim is True:
        _labels = tf.cast(tf.reshape(_labels, [-1, 128, 128, 1]), tf.float64)
        _pred = tf.cast(tf.reshape(_pred, [-1, 128, 128, 1]), tf.float64)
        # s = tf.image.ssim(_pred, _labels, max_val=1, filter_size=5, filter_sigma=1.5, k1=0.01, k2=0.03)
        s = tf.image.ssim_multiscale(_pred, _labels, max_val=1.0, filter_size=7, filter_sigma=1.5, k1=0.01, k2=0.03)
        ssim = (1.0 - tf.reduce_mean(s))
        # ssim = tf.Print(ssim, [ssim], message='SSIM = ')
        tf.compat.v1.add_to_collection('losses', ssim)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')
