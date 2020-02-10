import tensorflow as tf


NUM_CLASSES = 2


def loss_def(_logits, _labels):
    logits = tf.reshape(_logits, [-1, NUM_CLASSES])
    labels = tf.cast(tf.reshape(_labels, [-1]), tf.int32)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.compat.v1.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')
