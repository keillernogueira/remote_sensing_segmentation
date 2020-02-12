import os
import math
import argparse
import datetime
import random

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score

import tensorflow as tf

from config import *
from utils import *
from dataloaders.unique_image_loader import UniqueImageLoader
from networks.factory import model_factory
from networks.loss import loss_def


# def test(testing_data, testing_labels, testing_instances, batch_size, weight_decay, mean_full, std_full, update_type,
#          distribution_type, values, patch_acc_loss, patch_occur, model_name, former_model_path):
#     channels = testing_data.shape[-1]
#
#     # TEST NETWORK
#     crop = tf.compat.v1.placeholder(tf.int32)
#     x = tf.compat.v1.placeholder(tf.float32, [None, None])
#     y = tf.compat.v1.placeholder(tf.float32, [None, None])
#
#     keep_prob = tf.compat.v1.placeholder(tf.float32)  # dropout (keep probability)
#     is_training = tf.compat.v1.placeholder(tf.bool, [], name='is_training')
#
#     logits = model_factory(model_name, x, is_training, weight_decay, crop)
#
#     # Evaluate model
#     pred_up = tf.argmax(logits, dimension=3)
#
#     if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
#         crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
#                                            debug=True)
#     else:
#         crop_size = int(values[0])
#     stride_crop = int(math.floor(crop_size / 2.0))
#
#     # restore
#     saver_restore = tf.compat.v1.train.Saver()
#
#     all_cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
#     all_kappa = np.zeros((len(testing_data)), dtype=np.float32)
#     all_f1 = np.zeros((len(testing_data)), dtype=np.float32)
#
#     with tf.compat.v1.Session() as sess:
#         print('Model restored from ' + former_model_path + '!')
#         saver_restore.restore(sess, former_model_path)
#
#         for k in range(len(testing_data)):
#             # print testing_data[k].shape
#             h, w, c = testing_data[k].shape
#
#             instaces_stride_h = (
#                 int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
#                     ((h - crop_size) / stride_crop)) + 2)
#             instaces_stride_w = (
#                 int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
#                     ((w - crop_size) / stride_crop)) + 2)
#             instaces_stride = instaces_stride_h * instaces_stride_w
#             # print '--', instaces_stride, (instaces_stride/batch_size)
#             # ((instaces_stride/batch_size)+1 if instaces_stride%batch_size != 0 else (instaces_stride/batch_size))
#
#             prob_im = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
#             occur_im = np.zeros([h, w, NUM_CLASSES], dtype=np.uint32)
#
#             for i in range(0, (int(instaces_stride / batch_size) + 1 if instaces_stride % batch_size != 0 else int(
#                         instaces_stride / batch_size))):
#                 test_patches, test_classes, pos = create_patches_per_map(testing_data[k], testing_labels[k], crop_size,
#                                                                          stride_crop, i, batch_size)
#                 normalize_images(test_patches, mean_full, std_full)
#                 # raw_input_data("press")
#
#                 bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
#                 by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))
#
#                 _pred_up, _logits = sess.run([pred_up, logits], feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1.,
#                                                                            is_training: False})
#                 for j in range(len(_logits)):
#                     prob_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
#                     int(pos[j][1]):int(pos[j][1]) + crop_size, :] += _logits[j, :, :, :]
#                     occur_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
#                     int(pos[j][1]):int(pos[j][1]) + crop_size, :] += 1
#                     # index += 1
#
#             occur_im[np.where(occur_im == 0)] = 1
#             # np.save(output_path + 'prob_map' + str(testing_instances[k]) + '.npy', prob_im/occur_im.astype(float))
#             prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=2)
#             # print np.bincount(prob_im_argmax.astype(int).flatten())
#             # create_prediction_map(output_path+"predImg_"+testing_instances[k]+".jpeg",
#             # prob_im_argmax, testing_labels[k].shape)
#
#             cm_test_per_map = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
#             for t in range(h):
#                 for r in range(w):
#                     # print testing_labels[k][t][r]
#                     # print prob_im_argmax[t][r]
#                     cm_test_per_map[int(testing_labels[k, t, r])][int(prob_im_argmax[t, r])] += 1
#                     all_cm_test[int(testing_labels[k, t, r])][int(prob_im_argmax[t, r])] += 1
#
#             _sum = 0.0
#             total = 0
#             for i in range(len(cm_test_per_map)):
#                 _sum += (
#                     cm_test_per_map[i][i] / float(np.sum(cm_test_per_map[i])) if np.sum(cm_test_per_map[i]) != 0 else 0)
#                 total += cm_test_per_map[i][i]
#
#             cur_kappa = cohen_kappa_score(testing_labels[k].flatten(), prob_im_argmax.flatten())
#             cur_f1 = f1_score(testing_labels[k].flatten(), prob_im_argmax.flatten(), average='micro')
#             all_kappa[k] = cur_kappa
#             all_f1[k] = cur_f1
#
#             print(" -- Test Map " + testing_instances[k] + ": Overall Accuracy= " + str(total) +
#                   " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(cm_test_per_map))) +
#                   " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
#                   " F1 Score= " + "{:.4f}".format(cur_f1) +
#                   " Kappa= " + "{:.4f}".format(cur_kappa) +
#                   " Confusion Matrix= " + np.array_str(cm_test_per_map).replace("\n", "")
#                   )
#
#         _sum = 0.0
#         total = 0
#         for i in range(len(all_cm_test)):
#             _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
#             total += all_cm_test[i][i]
#
#         print(" -- Test ALL MAPS: Overall Accuracy= " + str(total) +
#               " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
#               " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
#               " F1 Score= " + np.array_str(all_f1).replace("\n", " ") +
#               " Mean F1 Score= " + "{:.6f}".format(np.sum(all_f1) / float(len(testing_data))) +
#               " Kappa= " + np.array_str(all_kappa).replace("\n", " ") +
#               " Mean Kappa Score= " + "{:.6f}".format(np.sum(all_kappa) / float(len(testing_data))) +
#               " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
#               )
#
#
# def validate_test(sess, testing_data, testing_labels, testing_instances, batch_size, mean_full, std_full, x, y, crop,
#                   keep_prob, is_training, pred_up, logits, crop_size, step, output_path):
#     stride_crop = int(math.floor(crop_size / 2.0))
#     all_cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
#     all_kappa = np.zeros((len(testing_data)), dtype=np.float32)
#     all_f1 = np.zeros((len(testing_data)), dtype=np.float32)
#     all_f1_per_class = np.zeros((NUM_CLASSES), dtype=np.float32)
#
#     for k in range(len(testing_data)):
#         # print testing_data[k].shape
#         h, w, c = testing_data[k].shape
#
#         instaces_stride_h = (int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
#             ((h - crop_size) / stride_crop)) + 2)
#         instaces_stride_w = (int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
#             ((w - crop_size) / stride_crop)) + 2)
#         instaces_stride = instaces_stride_h * instaces_stride_w
#         # print '--', instaces_stride, (instaces_stride/batch_size)
#         # ((instaces_stride/batch_size)+1 if instaces_stride%batch_size != 0 else (instaces_stride/batch_size))
#
#         prob_im = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
#         occur_im = np.zeros([h, w, NUM_CLASSES], dtype=np.uint32)
#
#         for i in range(0, (int(instaces_stride / batch_size) + 1 if instaces_stride % batch_size != 0 else int(
#                     instaces_stride / batch_size))):
#             test_patches, test_classes, pos = create_patches_per_map(testing_data[k], testing_labels[k], crop_size,
#                                                                      stride_crop, i, batch_size)
#             normalize_images(test_patches, mean_full, std_full)
#             # raw_input("press")
#
#             bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
#             by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))
#
#             _pred_up, _logits = sess.run([pred_up, logits],
#                                          feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1., is_training: False})
#             for j in range(len(_logits)):
#                 prob_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
#                         int(pos[j][1]):int(pos[j][1]) + crop_size, :] += _logits[j, :, :, :]
#                 occur_im[int(pos[j][0]):int(pos[j][0]) + crop_size, int(pos[j][1]):int(pos[j][1]) + crop_size, :] += 1
#                 # index += 1
#
#         occur_im[np.where(occur_im == 0)] = 1
#         # np.save(output_path + 'prob_map' + str(testing_instances[k]) + '.npy', prob_im/occur_im.astype(float))
#         prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=2)
#         # print np.bincount(prob_im_argmax.astype(int).flatten())
#         # create_prediction_map(output_path + "predImg_" + testing_instances[k] + "_step_" + str(step) + ".jpeg",
#         # prob_im_argmax, testing_labels[k].shape)
#
#         cm_test_per_map = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
#         for t in range(h):
#             for r in range(w):
#                 # print testing_labels[k][t][r]
#                 # print prob_im_argmax[t][r]
#                 if int(testing_labels[k][t][r]) != 6:  # not eroded
#                     cm_test_per_map[int(testing_labels[k][t, r])][int(prob_im_argmax[t, r])] += 1
#                     all_cm_test[int(testing_labels[k][t, r])][int(prob_im_argmax[t, r])] += 1
#
#         _sum = 0.0
#         total = 0
#         for i in range(len(cm_test_per_map)):
#             _sum += (
#                 cm_test_per_map[i][i] / float(np.sum(cm_test_per_map[i])) if np.sum(cm_test_per_map[i]) != 0 else 0)
#             total += cm_test_per_map[i][i]
#
#         cur_kappa = cohen_kappa_score(testing_labels[k][testing_labels[k] != 6],
#                                       prob_im_argmax[testing_labels[k] != 6])
#         cur_f1 = f1_score(testing_labels[k][testing_labels[k] != 6],
#                           prob_im_argmax[testing_labels[k] != 6], average='macro')
#         cur_f1_per_class = f1_score(testing_labels[k][testing_labels[k] != 6],
#                                     prob_im_argmax[testing_labels[k] != 6], average=None)
#         all_kappa[k] = cur_kappa
#         all_f1[k] = cur_f1
#         if len(cur_f1_per_class) == 5:  # in this case, the current image has no background class
#             cur_f1_per_class = np.append(cur_f1_per_class, 0.0)
#
#         all_f1_per_class += cur_f1_per_class
#
#         print("---- Iter " + str(step) +
#               " -- Test Map " + testing_instances[k] + ": Overall Accuracy= " + str(total) +
#               " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(cm_test_per_map))) +
#               " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
#               " F1 Score per class= " + np.array_str(cur_f1_per_class).replace("\n", "") +
#               " F1 Score= " + "{:.4f}".format(cur_f1) +
#               " Kappa= " + "{:.4f}".format(cur_kappa) +
#               " Confusion Matrix= " + np.array_str(cm_test_per_map).replace("\n", "")
#               )
#
#     _sum = 0.0
#     total = 0
#     for i in range(len(all_cm_test)):
#         _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
#         total += all_cm_test[i][i]
#
#     print("---- Iter " + str(step) +
#           " -- Test ALL MAPS: Overall Accuracy= " + str(total) +
#           " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
#           " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
#           " F1 Score= " + np.array_str(all_f1).replace("\n", " ") +
#           " Mean F1 Score= " + "{:.6f}".format(np.sum(all_f1) / float(len(testing_data))) +
#           " F1 Score per class= " + np.array_str(all_f1_per_class / float(len(testing_data))).replace("\n", "") +
#           " Kappa= " + np.array_str(all_kappa).replace("\n", " ") +
#           " Mean Kappa Score= " + "{:.6f}".format(np.sum(all_kappa) / float(len(testing_data))) +
#           " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
#           )
#
#
# def test_or_validate_whole_images(former_model_path, testing_data, testing_labels,
#                                   testing_instances, batch_size, weight_decay, mean_full, std_full, update_type,
#                                   distribution_type, model_name, values, output_path):
#     channels = testing_data[0].shape[-1]
#
#     bkp_values = values
#     for model in former_model_path:
#         # PLACEHOLDERS
#         crop = tf.compat.v1.placeholder(tf.int32)
#         x = tf.compat.v1.placeholder(tf.float32, [None, None])
#         y = tf.compat.v1.placeholder(tf.float32, [None, None])
#         keep_prob = tf.compat.v1.placeholder(tf.float32)
#         is_training = tf.compat.v1.placeholder(tf.bool, [], name='is_training')
#
#         logits = model_factory(model_name, x, is_training, weight_decay, crop)
#
#         pred_up = tf.argmax(logits, dimension=3)
#
#         # restore
#         saver_restore = tf.compat.v1.train.Saver()
#
#         with tf.compat.v1.Session() as sess:
#             print('Model restored from ' + model)
#             current_iter = int(model.split('-')[-1])
#             saver_restore.restore(sess, model)
#
#             if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or \
#                             distribution_type == 'multinomial':
#                 patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
#                 patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
#                 # patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
#                 values = bkp_values
#
#             # Evaluate model
#             if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or \
#                             distribution_type == 'multinomial':
#                 crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur,
#                                                    update_type, debug=True)
#             else:
#                 crop_size = int(values[0])
#
#             validate_test(sess, testing_data, testing_labels, testing_instances, batch_size,
#                           mean_full, std_full, x, y, crop, keep_prob, is_training, pred_up, logits, crop_size,
#                           current_iter, output_path)
#
#             if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or \
#                             distribution_type == 'multinomial':
#                 ind = np.where(values == crop_size)
#                 values = np.delete(values, ind)
#                 patch_acc_loss = np.delete(patch_acc_loss, ind)
#                 patch_occur = np.delete(patch_occur, ind)
#
#         tf.compat.v1.reset_default_graph()

def validation(sess, loader, batch_size, x, y, crop,
               keep_prob, is_training, pred_up, logits, step, crop_size):
    linear = np.arange(len(loader.test_distrib))
    all_cm_test = np.zeros((loader.num_classes, loader.num_classes), dtype=np.uint32)
    first = True

    for i in range(0, math.ceil(len(linear) / batch_size)):
        batch = linear[i * batch_size:min(i * batch_size + batch_size, len(linear))]
        test_patches, test_classes, test_masks = loader.dynamically_create_patches(loader.test_distrib[batch],
                                                                                   crop_size, is_train=False)
        loader.normalize_images(test_patches)

        bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
        by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))

        _pred_up, _logits = sess.run([pred_up, logits], feed_dict={x: bx, y: by, crop: crop_size,
                                                                   keep_prob: 1., is_training: False})

        if first is True:
            all_predcs = _pred_up
            all_labels = test_classes
            all_logits = _logits
            first = False
        else:
            all_predcs = np.concatenate((all_predcs, _pred_up))
            all_labels = np.concatenate((all_labels, test_classes))
            all_logits = np.concatenate((all_logits, _logits))

    # print(len(loader.test_distrib), all_labels.shape, all_predcs.shape, all_logits.shape)
    calc_accuracy_by_crop(all_labels, all_predcs, loader.num_classes, all_cm_test)

    _sum = 0.0
    total = 0
    for i in range(len(all_cm_test)):
        _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
        total += all_cm_test[i][i]

    cur_kappa = cohen_kappa_score(all_labels.flatten(), all_predcs.flatten())
    cur_f1 = f1_score(all_labels.flatten(), all_predcs.flatten(), average='macro')

    print("---- Iter " + str(step) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " -- Validation: Overall Accuracy= " + str(total) +
          " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(loader.num_classes)) +
          " F1 Score= " + "{:.4f}".format(cur_f1) +
          " Kappa= " + "{:.4f}".format(cur_kappa) +
          " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
          )

    return all_predcs, all_labels, all_logits


def train(loader, lr_initial, batch_size, niter,
          weight_decay, update_type, distribution_type, values,
          patch_acc_loss, patch_occur, patch_chosen_values, probs,
          output_path, model_name, former_model_path=None):

    # Network Parameters
    dropout = 0.5  # Dropout, probability to keep units

    # placeholders
    crop = tf.compat.v1.placeholder(tf.int32)
    x = tf.compat.v1.placeholder(tf.float32, [None, None])
    y = tf.compat.v1.placeholder(tf.float32, [None, None])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    is_training = tf.compat.v1.placeholder(tf.bool, [], name='is_training')

    logits = model_factory(model_name, x, is_training, weight_decay, crop, len(loader._mean), loader.num_classes)

    # Define loss and optimizer
    loss = loss_def(logits, y)

    global_step = tf.Variable(0, name='main_global_step', trainable=False)
    lr = tf.compat.v1.train.exponential_decay(lr_initial, global_step, 50000, 0.5, staircase=True)
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr,
                                                     momentum=0.9).minimize(loss, global_step=global_step)

    # Define Metric Evaluate model
    pred_up = tf.argmax(logits, axis=3)

    # Add ops to save and restore all the variables.
    saver = tf.compat.v1.train.Saver(max_to_keep=None)
    # restore
    saver_restore = tf.compat.v1.train.Saver()

    # Initializing the variables
    init = tf.compat.v1.global_variables_initializer()

    total_length = len(loader.train_distrib)
    shuffle = np.asarray(random.sample(range(total_length), total_length))
    epoch_counter = 1
    current_iter = 1

    # Launch the graph
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.compat.v1.Session() as sess:
        if former_model_path is not None and 'model' in former_model_path:
            current_iter = int(former_model_path.split('-')[-1])
            print('Model restored from ' + former_model_path)
            patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
            patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
            patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
            # print patch_acc_loss, patch_occur, patch_chosen_values
            saver_restore.restore(sess, former_model_path)
            current_iter = current_iter + 1
        else:
            sess.run(init)
            print('Model totally initialized!')

        # aux variables
        it = 0
        epoch_mean = 0.0
        epoch_cm_train = np.zeros((loader.num_classes, loader.num_classes), dtype=np.uint32)

        # Keep training until reach max iterations
        for step in range(current_iter, niter + 1):
            if distribution_type == 'multi_fixed':
                cur_size_int = np.random.randint(len(values))
                cur_patch_size = int(values[cur_size_int])
                print(cur_patch_size)
            elif distribution_type == 'uniform':
                cur_patch_size = int(np.random.uniform(values[0], values[-1] + 1, 1))
                cur_size_int = cur_patch_size - values[0]
                print(cur_patch_size)
            elif distribution_type == 'multinomial':
                cur_size_int = np.random.multinomial(1, probs).argmax()
                cur_patch_size = values[0] + cur_size_int
                print(cur_patch_size)
            elif distribution_type == 'single_fixed':
                cur_patch_size = int(values[0])

            # print 'new batch of crop size == ', cur_patch_size
            shuffle, batch, it = select_batch(shuffle, batch_size, it, total_length)

            b_x, b_y, b_mask = loader.dynamically_create_patches(loader.train_distrib[batch],
                                                                 cur_patch_size, is_train=True)
            loader.normalize_images(b_x)
            # print(b_x.shape, b_y.shape, b_mask.shape)

            # print b_x.shape, b_y.shape, b_mask.shape
            batch_x = np.reshape(b_x, (-1, cur_patch_size * cur_patch_size * b_x.shape[-1]))
            batch_y = np.reshape(b_y, (-1, cur_patch_size * cur_patch_size * 1))

            # Run optimization op (backprop)
            _, batch_loss, batch_pred_up = sess.run([optimizer, loss, pred_up],
                                                    feed_dict={x: batch_x, y: batch_y, crop: cur_patch_size,
                                                               keep_prob: dropout, is_training: True})

            acc, batch_cm_train = calc_accuracy_by_crop(b_y, batch_pred_up, loader.num_classes, epoch_cm_train, b_mask)
            epoch_mean += acc

            if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                    or distribution_type == 'multinomial':
                # print (batch_loss if update_type == 'loss' else (acc/float(np.sum(batch_cm_train))))
                patch_acc_loss[cur_size_int] += (
                    batch_loss * (epoch_counter / 10.0) if update_type == 'loss' else (
                        acc / float(np.sum(batch_cm_train))))
                # errorLoss[cur_size_int] += batch_loss*(epoch_counter/10.0)
                patch_occur[cur_size_int] += 1

            # DISPLAY TRAIN
            if step % DISPLAY_STEP == 0:
                _sum = 0.0
                for i in range(len(batch_cm_train)):
                    _sum += (batch_cm_train[i][i] / float(np.sum(batch_cm_train[i]))
                             if np.sum(batch_cm_train[i]) != 0 else 0)

                print("Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
                      " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                      " Absolut Right Pred= " + str(int(acc)) +
                      " Overall Accuracy= " + "{:.4f}".format(acc / float(np.sum(batch_cm_train))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(loader.num_classes)) +
                      " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                      )

            # DISPLAY TRAIN EPOCH
            if step % EPOCH_NUMBER == 0:
                _sum = 0.0
                for i in range(len(epoch_cm_train)):
                    _sum += (
                    epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i])) if np.sum(epoch_cm_train[i]) != 0 else 0)

                print("-- Iter " + str(step) + " -- Training Epoch:" +
                      " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(np.sum(epoch_cm_train))) +
                      " Normalized Accuracy= " + "{:.6f}".format(_sum / float(loader.num_classes)) +
                      " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                      )

                epoch_mean = 0.0
                epoch_cm_train = np.zeros((loader.num_classes, loader.num_classes), dtype=np.uint32)

            # DISPLAY VALIDATION
            if step % VAL_INTERVAL == 0:
                saver.save(sess, output_path + 'model', global_step=step)
                if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                        or distribution_type == 'multinomial':
                    np.save(output_path + 'patch_acc_loss_step_' + str(step) + '.npy', patch_acc_loss)
                    np.save(output_path + 'patch_occur_step_' + str(step) + '.npy', patch_occur)
                    np.save(output_path + 'patch_chosen_values_step_' + str(step) + '.npy', patch_chosen_values)
                    cur_patch_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur,
                                                           update_type, patch_chosen_values, debug=True)
                else:
                    cur_patch_val = int(values[0])

                validation(sess, loader, batch_size, x, y, crop,
                           keep_prob, is_training, pred_up, logits, step, cur_patch_val)

            # EPOCH IS COMPLETE
            if min(it + batch_size, total_length) == total_length or total_length == it + batch_size:
                epoch_counter += 1

        print("Optimization Finished!")

        # SAVE STATE
        saver.save(sess, output_path + 'model', global_step=step)
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            np.save(output_path + 'patch_acc_loss_step_' + str(step) + '.npy', patch_acc_loss)
            np.save(output_path + 'patch_occur_step_' + str(step) + '.npy', patch_occur)
            np.save(output_path + 'patch_chosen_values_step_' + str(step) + '.npy', patch_chosen_values)
            cur_patch_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                                   patch_chosen_values, debug=True)
        else:
            cur_patch_val = int(values[0])
        validation(sess, loader, batch_size, x, y, crop,
                   keep_prob, is_training, pred_up, logits, step, cur_patch_val)

    tf.compat.v1.reset_default_graph()


def generate_final_maps(former_model_path, loader,
                        batch_size, weight_decay,
                        update_type, distribution_type, model_name, values, output_path):
    # PLACEHOLDERS
    crop = tf.compat.v1.placeholder(tf.int32)
    x = tf.compat.v1.placeholder(tf.float32, [None, None])
    y = tf.compat.v1.placeholder(tf.float32, [None, None])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    is_training = tf.compat.v1.placeholder(tf.bool, [], name='is_training')

    logits = model_factory(model_name, x, is_training, weight_decay, crop, len(loader._mean), loader.num_classes)

    pred_up = tf.argmax(logits, axis=3)

    # restore
    saver_restore = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        current_iter = int(former_model_path.split('-')[-1])
        print('Model restored from ' + former_model_path)
        patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
        patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
        # patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
        saver_restore.restore(sess, former_model_path)

        # Evaluate model
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                               debug=True)
        else:
            crop_size = int(values[0])
        # stride_crop = int(math.floor(crop_size / 2.0))

        _, _, all_logits = validation(sess, loader, batch_size, x, y, crop,
                                      keep_prob, is_training, pred_up, logits, current_iter, crop_size)

        prob_im = np.zeros([loader.labels.shape[0], loader.labels.shape[1], loader.num_classes], dtype=np.float32)
        occur_im = np.zeros([loader.labels.shape[0], loader.labels.shape[1], loader.num_classes], dtype=np.float32)
        for i in range(len(loader.test_distrib)):
            cur_x = loader.test_distrib[i][0]
            cur_y = loader.test_distrib[i][1]
            prob_im[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += all_logits[i, :, :, :]
            occur_im[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += 1

        occur_im[np.where(occur_im == 0)] = 1
        # np.save(output_path + 'prob_map' + str(testing_instances[k]) + '.npy', prob_im/occur_im.astype(float))
        prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=2)



        create_prediction_map(output_path + os.listdir(loader.dataset_input_path)[0].split('_')[0] +
                              '_prediction', prob_im_argmax)

    tf.compat.v1.reset_default_graph()


'''
RUN:
python dynamic.py /home/users/keiller/tcu/temp_dataset/ /home/users/keiller/tcu/tgrs/output/fold1/ /home/users/keiller/tcu/tgrs/output/fold1/ 1 0.01 0.001 128 150000 100 75 dilated_grsl_rate8 multi_fixed 100,200,300,400 acc training
python dynamic.py /home/users/keiller/tcu/temp_dataset/ /home/users/keiller/tcu/tgrs/output/fold3/ /home/users/keiller/tcu/tgrs/output/fold3/model-120000 3 0.01 0.001 32 150000 75 50 dilated8_grsl multi_fixed 50,75,100 acc training > /home/users/keiller/tcu/tgrs/output/fold3/out_v1.txt
CUDA_VISIBLE_DEVICES=2 python3 main.py --operation training --output_path /home/kno/remote_sensing_segmentation/output/ --dataset_input_path /home/kno/dataset_laranjal/Dataset_Laranjal/Parrot\ Sequoia/ --dataset_gt_path /home/kno/dataset_laranjal/Dataset_Laranjal/Arvore_Segmentacao\ \(Sequoia\)/sequoia_raster.tif --num_classes 2 --model_name dilated_grsl_rate8 --values 25,50
'''


def main():
    parser = argparse.ArgumentParser(description='main')
    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation [Options: training | validate_test | generate_map]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--dataset_input_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--dataset_gt_path', type=str, required=True, help='Ground truth path.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes.')
    parser.add_argument('--dataset_split_method', type=str, default='train_test',
                        help='Split method the dataset [Options: train_test]')

    # model options
    parser.add_argument('--model_name', type=str, required=True, default='mobilefacenet',
                        help='Model to test [Options: dilated_grsl_rate8]')
    parser.add_argument('--model_path', type=str, default=None, help='Model path.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--niter', type=int, default=50000, help='Number of iterations')

    # dynamic dialted convnet options
    parser.add_argument('--reference_crop_size', type=int, default=25, help='Reference crop size.')
    parser.add_argument('--reference_stride_crop', type=int, default=5, help='Reference crop stride')
    parser.add_argument('--distribution_type', type=str, default='multi_fixed',
                        help='Distribution type [Options: uniform, multi_fixed, multinomial]')
    parser.add_argument('--values', type=str, default=None, help='Values considered in the distribution.')
    parser.add_argument('--update_type', type=str, default='acc', help='Update type [Options: loss, acc]')

    args = parser.parse_args()
    args.values = [int(i) for i in args.values.split(',')]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)

    if args.distribution_type == 'multi_fixed':
        patch_acc_loss = np.zeros(len(args.values), dtype=np.float32)
        patch_occur = np.zeros(len(args.values), dtype=np.int32)
        patch_chosen_values = np.zeros(len(args.values), dtype=np.int32)
    elif args.distribution_type == 'uniform' or args.distribution_type == 'multinomial':
        patch_acc_loss = np.zeros(args.values[-1] - args.values[0] + 1, dtype=np.float32)
        patch_occur = np.zeros(args.values[-1] - args.values[0] + 1, dtype=np.int32)
        patch_chosen_values = np.zeros(args.values[-1] - args.values[0] + 1, dtype=np.int32)
        probs = define_multinomial_probs(args.values)

    loader = UniqueImageLoader(args.dataset_input_path, args.dataset_gt_path, args.num_classes, args.output_path)
    print(loader.data.shape, loader.labels.shape)
    loader.split_dataset(args.reference_crop_size, args.reference_stride_crop, args.dataset_split_method)
    print(len(loader.train_distrib), len(loader.test_distrib))
    loader.create_or_load_mean(args.reference_crop_size, args.reference_stride_crop)

    if args.operation == 'training':
        train(loader, args.learning_rate, args.batch_size, args.niter,
              args.weight_decay, args.update_type, args.distribution_type, args.values,
              (None if args.distribution_type == 'single_fixed' else patch_acc_loss),
              (None if args.distribution_type == 'single_fixed' else patch_occur),
              (None if args.distribution_type == 'single_fixed' else patch_chosen_values),
              (None if args.distribution_type != 'multinomial' else probs),
              args.output_path, args.model_name, args.model_path)
    # elif args.operation == 'validate_test':
    #     test_or_validate_whole_images(args.model_path.split(","), loader,
    #                                   args.batch_size, args.weight_decay, args.update_type,
    #                                   args.distribution_type, args.model_name,
    #                                   np.asarray(args.values), args.output_path)
    elif args.operation == 'generate_map':
        generate_final_maps(args.model_path, loader,
                            args.batch_size, args.weight_decay, args.update_type,
                            args.distribution_type, args.model_name, args.values, args.output_path)
    else:
        raise NotImplementedError("Process " + args.operation + "not found!")


if __name__ == "__main__":
    main()
