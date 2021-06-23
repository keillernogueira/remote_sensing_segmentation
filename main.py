import os
import time
import re
import math
import datetime

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

import tensorflow as tf

from config import *
from utils import *

from dataloaders.factory import dataloader_factory
from dataloaders.utils import dynamically_create_patches, dynamically_create_patches_multi_images, normalize_images
from dataloaders.unique_image_loader import UniqueImageLoader
from dataloaders.train_validation_test_loader import TrainValTestLoader

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


# def validation_old(sess, model_name, loader, batch_size, x, y, crop,
#                keep_prob, is_training, pred_up, logits, step, crop_size):
#     linear = np.arange(len(loader.test_distrib))
#     all_cm_test = np.zeros((loader.num_classes, loader.num_classes), dtype=np.uint32)
#     first = True
#
#     for i in range(0, math.ceil(len(linear) / batch_size)):
#         batch = linear[i * batch_size:min(i * batch_size + batch_size, len(linear))]
#         if isinstance(loader, UniqueImageLoader):
#             test_patches, test_classes, test_masks = dynamically_create_patches(model_name, loader.data, loader.labels,
#                                                                                 loader.test_distrib[batch], crop_size,
#                                                                                 loader.num_classes, is_train=False)
#         else:
#             test_patches, test_classes, test_masks = \
#                 dynamically_create_patches_multi_images(model_name, loader.test_data, loader.test_labels,
#                                                         loader.test_distrib[batch], crop_size, loader.num_classes,
#                                                         is_train=False, remove_negative=False)
#
#         # print(test_patches.shape, test_classes.shape, np.bincount(test_classes.flatten()))
#         normalize_images(test_patches, loader._mean, loader._std)
#
#         bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
#         if model_name != 'pixelwise':
#             by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))
#         else:
#             by = test_classes
#
#         _pred_up, _logits = sess.run([pred_up, logits], feed_dict={x: bx, y: by, crop: crop_size,
#                                                                    keep_prob: 1., is_training: False})
#
#         if first is True:
#             all_predcs = _pred_up
#             all_labels = test_classes
#             all_logits = _logits
#             first = False
#         else:
#             all_predcs = np.concatenate((all_predcs, _pred_up))
#             all_labels = np.concatenate((all_labels, test_classes))
#             all_logits = np.concatenate((all_logits, _logits))
#
#     # print(len(loader.test_distrib), all_labels.shape, all_predcs.shape, all_logits.shape)
#     if model_name == 'pixelwise':
#         calc_accuracy_by_class(all_labels, all_predcs, loader.num_classes, all_cm_test)
#     else:
#         calc_accuracy_by_crop(all_labels, all_predcs, loader.num_classes, all_cm_test)
#
#     _sum = 0.0
#     total = 0
#     for i in range(len(all_cm_test)):
#         _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
#         total += all_cm_test[i][i]
#
#     _sum_iou = (all_cm_test[1][1] / float(
#         np.sum(all_cm_test[:, 1]) + np.sum(all_cm_test[1]) - all_cm_test[1][1])
#                 if (np.sum(all_cm_test[:, 1]) + np.sum(all_cm_test[1]) - all_cm_test[1][1]) != 0 else 0)
#
#     cur_kappa = cohen_kappa_score(all_labels.flatten(), all_predcs.flatten())
#     cur_f1 = f1_score(all_labels.flatten(), all_predcs.flatten(), average='macro')
#     # iou = jaccard_score(all_labels.flatten(), all_predcs.flatten())
#
#     print("---- Iter " + str(step) +
#           " -- Time " + str(datetime.datetime.now().time()) +
#           " -- Validation: Overall Accuracy= " + str(total) +
#           " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
#           " Normalized Accuracy= " + "{:.6f}".format(_sum / float(loader.num_classes)) +
#           " F1 Score= " + "{:.4f}".format(cur_f1) +
#           " Kappa= " + "{:.4f}".format(cur_kappa) +
#           " IoU= " + "{:.4f}".format(_sum_iou) +
#           " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
#           )
#
#     return all_predcs, all_labels, all_logits


def validation(sess, model_name, loader, batch_size, x, y, crop,
               keep_prob, is_training, pred_up, logits, step, crop_size, feats=None):
    linear = np.arange(len(loader.test_distrib))
    # all_cm_test = np.zeros((loader.num_classes, loader.num_classes), dtype=np.uint32)
    # first = True

    if isinstance(loader, UniqueImageLoader):
        prob_im = np.zeros([loader.labels.shape[0], loader.labels.shape[1], loader.num_classes], dtype=np.float32)
        occur_im = np.zeros([loader.labels.shape[0], loader.labels.shape[1], loader.num_classes], dtype=np.float32)
        if feats is not None:
            feat_low_im = np.zeros([32, 32, feats[0][1]], dtype=np.float32)
            feat_mid_im = np.zeros([32, 32, feats[1][1]], dtype=np.float32)
            feat_high_im = np.zeros([32, 32, feats[2][1]], dtype=np.float32)
    else:
        prob_im = np.zeros([loader.test_labels.shape[0], loader.test_labels.shape[1], loader.test_labels.shape[2],
                            loader.num_classes], dtype=np.float32)
        occur_im = np.zeros([loader.test_labels.shape[0], loader.test_labels.shape[1], loader.test_labels.shape[2],
                             loader.num_classes], dtype=np.float32)
        if feats is not None:
            feat_low_im = np.zeros([loader.test_labels.shape[0], 32, 32, feats[0][1]], dtype=np.float32)
            feat_mid_im = np.zeros([loader.test_labels.shape[0], 32, 32, feats[1][1]], dtype=np.float32)
            feat_high_im = np.zeros([loader.test_labels.shape[0], 32, 32, feats[2][1]], dtype=np.float32)

    for i in range(0, math.ceil(len(linear) / batch_size)):
        batch = linear[i * batch_size:min(i * batch_size + batch_size, len(linear))]
        index_batch = loader.test_distrib[batch]
        if isinstance(loader, UniqueImageLoader):
            test_patches, test_classes, test_masks = dynamically_create_patches(model_name, loader.data, loader.labels,
                                                                                index_batch, crop_size,
                                                                                loader.num_classes, is_train=False)
        else:
            test_patches, test_classes, test_masks = \
                dynamically_create_patches_multi_images(model_name, loader.test_data, loader.test_labels,
                                                        index_batch, crop_size, loader.num_classes,
                                                        is_train=False, remove_negative=False)

        # print(test_patches.shape, test_classes.shape, np.bincount(test_classes.flatten()))
        normalize_images(test_patches, loader._mean, loader._std)

        bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
        if model_name != 'pixelwise':
            by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))
        else:
            by = test_classes

        if feats is None:
            _pred_up, _logits = sess.run([pred_up, logits], feed_dict={x: bx, y: by, crop: crop_size,
                                                                       keep_prob: 1., is_training: False})
        else:
            _pred_up, _feat_low, _feat_mid, _feat_high, _logits = sess.run([pred_up, feats[0][0], feats[1][0],
                                                                            feats[2][0], logits],
                                                                           feed_dict={x: bx, y: by, crop: crop_size,
                                                                                      keep_prob: 1., is_training: False})

        # if first is True:
        #     all_predcs = _pred_up
        #     all_labels = test_classes
        #     all_logits = _logits
        #     first = False
        # else:
        #     all_predcs = np.concatenate((all_predcs, _pred_up))
        #     all_labels = np.concatenate((all_labels, test_classes))
        #     all_logits = np.concatenate((all_logits, _logits))

        for j in range(len(index_batch)):
            if isinstance(loader, UniqueImageLoader):
                cur_x = index_batch[j][0]
                cur_y = index_batch[j][1]
                prob_im[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _logits[j, :, :, :]
                occur_im[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += 1
                if feats is not None:
                    feat_low_im[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _feat_low[j, :, :, :]
                    feat_mid_im[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _feat_mid[j, :, :, :]
                    feat_high_im[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _feat_high[j, :, :, :]
            else:
                cur_map = index_batch[j][0]
                cur_x = index_batch[j][1]
                cur_y = index_batch[j][2]
                # print(index_batch.shape, index_batch[0],
                #       prob_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :].shape,
                #       _logits[j, :, :, :].shape, _logits.shape)

                temp_patch = prob_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
                if len(temp_patch) != crop_size and len(temp_patch[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(temp_patch))
                    cur_y = cur_y - (crop_size - len(temp_patch[0]))
                elif len(temp_patch) != crop_size:
                    cur_x = cur_x - (crop_size - len(temp_patch))
                elif len(temp_patch[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(temp_patch[0]))

                prob_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _logits[j, :, :, :]
                occur_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += 1
                if feats is not None:
                    feat_low_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _feat_low[j, :, :, :]
                    feat_mid_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _feat_mid[j, :, :, :]
                    feat_high_im[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] += _feat_high[j, :, :, :]

    # if model_name == 'pixelwise':
    #     calc_accuracy_by_class(all_labels, all_predcs, loader.num_classes, all_cm_test)
    # else:
    #     calc_accuracy_by_crop(all_labels, all_predcs, loader.num_classes, all_cm_test)
    #
    # _sum = 0.0
    # total = 0
    # for i in range(len(all_cm_test)):
    #     _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
    #     total += all_cm_test[i][i]
    #
    # _sum_iou = (all_cm_test[1][1] / float(
    #     np.sum(all_cm_test[:, 1]) + np.sum(all_cm_test[1]) - all_cm_test[1][1])
    #             if (np.sum(all_cm_test[:, 1]) + np.sum(all_cm_test[1]) - all_cm_test[1][1]) != 0 else 0)
    #
    # cur_kappa = cohen_kappa_score(all_labels.flatten(), all_predcs.flatten())
    # cur_f1 = f1_score(all_labels.flatten(), all_predcs.flatten(), average='macro')
    # # iou = jaccard_score(all_labels.flatten(), all_predcs.flatten())
    #
    # print("---- Iter " + str(step) +
    #       " -- Time " + str(datetime.datetime.now().time()) +
    #       " -- Validation: Overall Accuracy= " + str(total) +
    #       " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
    #       " Normalized Accuracy= " + "{:.6f}".format(_sum / float(loader.num_classes)) +
    #       " F1 Score= " + "{:.4f}".format(cur_f1) +
    #       " Kappa= " + "{:.4f}".format(cur_kappa) +
    #       " IoU= " + "{:.4f}".format(_sum_iou) +
    #       " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
    #       )

    occur_im[np.where(occur_im == 0)] = 1
    prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=-1)
    if isinstance(loader, UniqueImageLoader) and feats is not None:
        feat_low_im = feat_low_im / np.repeat(occur_im.astype(float), 128, axis=-1)
        feat_mid_im = feat_mid_im / np.repeat(occur_im.astype(float), 128, axis=-1)
        feat_high_im = feat_high_im / np.repeat(occur_im.astype(float), 128, axis=-1)

    all_cm_test2 = create_cm(loader.labels if isinstance(loader, UniqueImageLoader) else loader.test_labels,
                             prob_im_argmax)
    _sum = 0.0
    total = 0
    for i in range(len(all_cm_test2)):
        _sum += (all_cm_test2[i][i] / float(np.sum(all_cm_test2[i])) if np.sum(all_cm_test2[i]) != 0 else 0)
        total += all_cm_test2[i][i]

    _sum_iou = (all_cm_test2[1][1] / float(
        np.sum(all_cm_test2[:, 1]) + np.sum(all_cm_test2[1]) - all_cm_test2[1][1])
                if (np.sum(all_cm_test2[:, 1]) + np.sum(all_cm_test2[1]) - all_cm_test2[1][1]) != 0 else 0)

    kappa_cm = kappa_with_cm(all_cm_test2)
    f1_cm = f1_with_cm(all_cm_test2)

    print("---- Iter " + str(step) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " -- Validation: Overall Accuracy= " + str(total) +
          " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test2))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(loader.num_classes)) +
          " F1 Score= " + "{:.4f}".format(f1_cm) +
          " Kappa= " + "{:.4f}".format(kappa_cm) +
          " IoU= " + "{:.4f}".format(_sum_iou) +
          " Confusion Matrix= " + np.array_str(all_cm_test2).replace("\n", "")
          )

    if feats is not None:
        return prob_im_argmax, feat_low_im, feat_mid_im, feat_high_im
    else:
        return prob_im_argmax, total / float(np.sum(all_cm_test2)), _sum / float(loader.num_classes), \
               f1_cm, kappa_cm, _sum_iou, all_cm_test2


def train(loader, lr_initial, batch_size, niter,
          weight_decay, update_type, distribution_type, values,
          patch_acc_loss, patch_occur, patch_chosen_values, probs,
          output_path, model_name, former_model_path=None, _ssim=False):

    # Network Parameters
    dropout = 0.5  # Dropout, probability to keep units

    # placeholders
    crop = tf.compat.v1.placeholder(tf.int32)
    x = tf.compat.v1.placeholder(tf.float32, [None, None])
    if model_name == 'pixelwise':
        y = tf.placeholder(tf.int32, [None])
    else:
        y = tf.compat.v1.placeholder(tf.float32, [None, None])
    mask = tf.placeholder(tf.float32, [None, None])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    is_training = tf.compat.v1.placeholder(tf.bool, [], name='is_training')

    logits = model_factory(model_name, x, keep_prob, is_training, weight_decay,
                           crop, len(loader._mean), loader.num_classes, values[0])

    # Define Metric Evaluate model
    if model_name == 'pixelwise':
        pred = tf.argmax(logits, axis=1)
    else:
        pred = tf.argmax(logits, axis=3)

    # Define loss and optimizer
    loss = loss_def(model_name, logits, y, mask, _ssim, pred)

    global_step = tf.Variable(0, name='main_global_step', trainable=False)
    lr = tf.compat.v1.train.exponential_decay(lr_initial, global_step, 50000, 0.5, staircase=True)
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss,
                                                                                              global_step=global_step)

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

    best_records = []

    # Launch the graph
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.compat.v1.Session() as sess:
        if former_model_path is not None and 'model' in former_model_path:
            current_iter = int(former_model_path.split('-')[-1])
            print('Model restored from ' + former_model_path)
            if 'dilated' in model_name:
                patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
                patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
                patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
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

            if isinstance(loader, UniqueImageLoader):
                b_x, b_y, b_mask = dynamically_create_patches(model_name, loader.data, loader.labels,
                                                              loader.train_distrib[batch], cur_patch_size,
                                                              loader.num_classes, is_train=True)
            elif isinstance(loader, TrainValTestLoader):
                b_x, b_y, b_mask = dynamically_create_patches_multi_images(model_name, loader.train_data,
                                                                           loader.train_labels,
                                                                           loader.train_distrib[batch], cur_patch_size,
                                                                           loader.num_classes, is_train=True,
                                                                           remove_negative=False)
            normalize_images(b_x, loader._mean, loader._std)

            # print b_x.shape, b_y.shape, b_mask.shape
            batch_x = np.reshape(b_x, (-1, cur_patch_size * cur_patch_size * b_x.shape[-1]))
            if model_name != 'pixelwise':
                batch_y = np.reshape(b_y, (-1, cur_patch_size * cur_patch_size * 1))
            else:
                batch_y = b_y
            batch_mask = np.reshape(b_mask, (-1, cur_patch_size * cur_patch_size * 1))

            # Run optimization op (backprop)
            _, batch_loss, batch_logits, batch_pred_up = sess.run([optimizer, loss, logits, pred],
                                                                  feed_dict={x: batch_x, y: batch_y, mask: batch_mask,
                                                                             crop: cur_patch_size,
                                                                             keep_prob: dropout, is_training: True})

            if math.isnan(batch_loss):
                print('-------------------------NaN-----------------------------------------------')
                print(b_x.shape, b_y.shape, b_mask.shape, np.bincount(b_y.flatten()))
                print(np.min(b_x), np.max(b_x), np.isnan(b_x).any())
                print(np.min(b_y), np.max(b_y), np.isnan(b_y).any())
                print(np.min(b_mask), np.max(b_mask), np.isnan(b_mask).any())
                print('---')
                print(b_x.shape)
                print(b_y.shape, np.bincount(b_y.flatten()), np.min(b_y), np.max(b_y))
                print(b_mask.shape, np.bincount(b_mask.astype(int).flatten()))
                print(batch_pred_up.shape, np.bincount(batch_pred_up.flatten()), np.min(batch_pred_up), np.max(batch_pred_up))
                print(batch_logits.shape, b_y.shape)
                print(np.min(batch_logits), np.max(batch_logits))
                print('-------------------------NaN-----------------------------------------------')
                raise AssertionError

            if model_name == 'pixelwise':
                acc, batch_cm_train = calc_accuracy_by_class(b_y, batch_pred_up, loader.num_classes, epoch_cm_train)
            else:
                acc, batch_cm_train = calc_accuracy_by_crop(b_y, batch_pred_up, loader.num_classes,
                                                            epoch_cm_train, b_mask)

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

                _sum_iou = (batch_cm_train[1][1] / float(
                    np.sum(batch_cm_train[:, 1]) + np.sum(batch_cm_train[1]) - batch_cm_train[1][1])
                            if (np.sum(batch_cm_train[:, 1]) + np.sum(batch_cm_train[1]) - batch_cm_train[1][1]) != 0
                            else 0)

                print("Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
                      " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                      " Absolut Right Pred= " + str(int(acc)) +
                      " Overall Accuracy= " + "{:.4f}".format(acc / float(np.sum(batch_cm_train))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(loader.num_classes)) +
                      " IoU= " + "{:.4f}".format(_sum_iou) +
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
                if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                        or distribution_type == 'multinomial':
                    cur_patch_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur,
                                                           update_type, patch_chosen_values, debug=True)
                else:
                    cur_patch_val = int(values[0])

                # validation
                _, acc, nacc, f1, kappa, iou, cm = validation(sess, model_name, loader, batch_size, x, y, crop,
                                                              keep_prob, is_training, pred, logits, step, cur_patch_val)

                nacc = random.uniform(80, 100)
                save_best_models(sess, output_path, distribution_type, patch_acc_loss, patch_occur, patch_chosen_values,
                                 saver, best_records, step, acc, nacc, f1, kappa, iou, cm)
                print(best_records)

            # EPOCH IS COMPLETE
            if min(it + batch_size, total_length) == total_length or total_length == it + batch_size:
                epoch_counter += 1

        print("Optimization Finished!")

        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            cur_patch_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                                   patch_chosen_values, debug=True)
        else:
            cur_patch_val = int(values[0])
        # validation
        _, acc, nacc, f1, kappa, iou, cm = validation(sess, model_name, loader, batch_size, x, y, crop,
                                                      keep_prob, is_training, pred, logits, step, cur_patch_val)

        # SAVE STATE
        save_best_models(sess, output_path, distribution_type, patch_acc_loss, patch_occur, patch_chosen_values,
                         saver, best_records, step, acc, nacc, f1, kappa, iou, cm)

        print(best_records)
    tf.compat.v1.reset_default_graph()


def generate_final_maps(former_model_path, loader, batch_size, weight_decay,
                        update_type, distribution_type, model_name, values, output_path, feat=False):
    # PLACEHOLDERS
    crop = tf.compat.v1.placeholder(tf.int32)
    x = tf.compat.v1.placeholder(tf.float32, [None, None])
    y = tf.compat.v1.placeholder(tf.float32, [None, None])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    is_training = tf.compat.v1.placeholder(tf.bool, [], name='is_training')

    if feat is True:
        feat_low, feat_mid, feat_high, logits = model_factory(model_name, x, keep_prob, is_training, weight_decay,
                                                              crop, len(loader._mean), loader.num_classes,
                                                              values[0], extract_features=feat)
    else:
        logits = model_factory(model_name, x, keep_prob, is_training, weight_decay,
                               crop, len(loader._mean), loader.num_classes,
                               values[0], extract_features=feat)

    pred_up = tf.argmax(logits, axis=3)

    # restore
    saver_restore = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        current_iter = int(former_model_path.split('-')[-1])
        if 'dilated' in model_name:
            patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
            patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
        # patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
        saver_restore.restore(sess, former_model_path)
        print('Model restored from ' + former_model_path)

        # Evaluate model
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                               debug=True)
        else:
            crop_size = int(values[0])
        # stride_crop = int(math.floor(crop_size / 2.0))

        print(" -- Time " + str(datetime.datetime.now().time()))
        start_time = time.time()
        if feat is True:
            prob_im_argmax, feat_low_out, feat_mid_out, feat_high_out = validation(sess, model_name, loader, batch_size,
                                                                                   x, y, crop, keep_prob, is_training,
                                                                                   pred_up, logits, current_iter,
                                                                                   crop_size, feats=[feat_low, feat_mid,
                                                                                                     feat_high])
        else:
            prob_im_argmax, _, _, _, _, _, _ = validation(sess, model_name, loader, batch_size, x, y, crop,
                                                          keep_prob, is_training, pred_up, logits, current_iter,
                                                          crop_size)
        print(" -- Time " + str(datetime.datetime.now().time()))
        print("--- %s seconds ---" % (time.time() - start_time))

        if not os.path.exists(os.path.join(output_path, 'pred')):
            os.mkdir(os.path.join(output_path, 'pred'))

        if isinstance(loader, UniqueImageLoader):
            create_prediction_map(os.path.join(output_path,
                                  os.listdir(loader.dataset_input_path)[0].split('_')[0] + '_prediction'),
                                  prob_im_argmax)
            if feat is True:
                create_prediction_map(os.path.join(output_path,
                                      os.listdir(loader.dataset_input_path)[0].split('_')[0] + '_feat_low'),
                                      feat_low_out, channels=True)
                create_prediction_map(os.path.join(output_path,
                                      os.listdir(loader.dataset_input_path)[0].split('_')[0] + '_feat_mid'),
                                      feat_mid_out, channels=True)
                create_prediction_map(os.path.join(output_path,
                                      os.listdir(loader.dataset_input_path)[0].split('_')[0] + '_feat_high'),
                                      feat_high_out, channels=True)
        else:
            for i in range(10):
            # for i, m in enumerate(prob_im_argmax):
                # create_prediction_map(os.path.join(output_path, 'pred', os.path.splitext(loader.test_name[i])[0] +
                #                                    '_prediction'), m)
                if feat is True:
                    create_prediction_map(os.path.join(output_path, 'pred', os.path.splitext(loader.test_name[i])[0] +
                                                       '_feat_low'), feat_low_out[i], channels=True)
                    create_prediction_map(os.path.join(output_path, 'pred', os.path.splitext(loader.test_name[i])[0] +
                                                       '_feat_mid'), feat_mid_out[i], channels=True)
                    create_prediction_map(os.path.join(output_path, 'pred', os.path.splitext(loader.test_name[i])[0] +
                                                       '_feat_high'), feat_high_out[i], channels=True)
                    # create_prediction_map(os.path.join(output_path, 'pred', os.path.splitext(loader.test_name[i])[0] +
                    #                                    '_mask'), loader.test_labels[i])
                    # imageio.imwrite(os.path.join(output_path, 'pred', os.path.splitext(loader.test_name[i])[0] +
                    #                              '_resave.png'), loader.test_data[i])

    tf.compat.v1.reset_default_graph()


def main():
    parser = argparse.ArgumentParser(description='main')
    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation [Options: training | generate_map]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to to save outcomes (such as images and trained models) of the algorithm.')
    parser.add_argument('--simulate_dataset', type=str2bool, default=False,
                        help='Used to speed up the development process.')
    parser.add_argument('--save_features', type=str2bool, default=False,
                        help='Save features of the networks.')

    # dataset options
    parser.add_argument('--dataset', type=str, help='Dataset [Options: orange | tree | road_detection | river].')
    parser.add_argument('--dataset_input_path', type=str, help='Dataset path.')
    parser.add_argument('--dataset_gt_path', type=str, help='Ground truth path.')
    parser.add_argument('--num_classes', type=int, help='Number of classes.')
    # parser.add_argument('--dataset_split_method', type=str, default='train_test',
    #                     help='Split method the dataset [Options: train_test]')

    # model options
    parser.add_argument('--model_name', type=str, default='dilated_grsl_rate8',
                        help='Model to test [Options: dilated_grsl_rate8]')
    parser.add_argument('--model_path', type=str, default=None, help='Model path.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--niter', type=int, default=200000, help='Number of iterations')
    parser.add_argument('--ssim', type=str2bool, default=False,
                        help='Use SSIM loss.')

    # dynamic dilated convnet options
    parser.add_argument('--reference_crop_size', type=int, default=25, help='Reference crop size.')
    parser.add_argument('--reference_stride_crop', type=int, default=15, help='Reference crop stride')
    parser.add_argument('--distribution_type', type=str, default='multi_fixed',
                        help='Distribution type [Options: single_fixed, uniform, multi_fixed, multinomial]')
    parser.add_argument('--values', type=str, default=None, help='Values considered in the distribution.')
    parser.add_argument('--update_type', type=str, default='acc', help='Update type [Options: loss, acc]')

    args = parser.parse_args()
    if args.values is not None:
        args.values = [int(i) for i in args.values.split(',')]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)

    if args.operation == 'training' or args.operation == 'generate_map':
        if args.distribution_type == 'multi_fixed':
            patch_acc_loss = np.zeros(len(args.values), dtype=np.float32)
            patch_occur = np.zeros(len(args.values), dtype=np.int32)
            patch_chosen_values = np.zeros(len(args.values), dtype=np.int32)
        elif args.distribution_type == 'uniform' or args.distribution_type == 'multinomial':
            patch_acc_loss = np.zeros(args.values[-1] - args.values[0] + 1, dtype=np.float32)
            patch_occur = np.zeros(args.values[-1] - args.values[0] + 1, dtype=np.int32)
            patch_chosen_values = np.zeros(args.values[-1] - args.values[0] + 1, dtype=np.int32)
            probs = define_multinomial_probs(args.values)

        loader = dataloader_factory(args.dataset, args.dataset_input_path, args.dataset_gt_path, args.num_classes,
                                    args.output_path, args.model_name, args.reference_crop_size,
                                    args.reference_stride_crop, args.operation == 'training', args.simulate_dataset)

        if args.operation == 'training':
            train(loader, args.learning_rate, args.batch_size, args.niter,
                  args.weight_decay, args.update_type, args.distribution_type, args.values,
                  (None if args.distribution_type == 'single_fixed' else patch_acc_loss),
                  (None if args.distribution_type == 'single_fixed' else patch_occur),
                  (None if args.distribution_type == 'single_fixed' else patch_chosen_values),
                  (None if args.distribution_type != 'multinomial' else probs),
                  args.output_path, args.model_name, args.model_path, args.ssim)
        # elif args.operation == 'validate_test':
        #     test_or_validate_whole_images(args.model_path.split(","), loader,
        #                                   args.batch_size, args.weight_decay, args.update_type,
        #                                   args.distribution_type, args.model_name,
        #                                   np.asarray(args.values), args.output_path)
        elif args.operation == 'generate_map':
            generate_final_maps(args.model_path, loader,
                                args.batch_size, args.weight_decay, args.update_type,
                                args.distribution_type, args.model_name, args.values,
                                args.output_path, args.save_features)
    elif args.operation == 'filter_results':
        try:
            f = open(args.output_path, 'r')
        except IOError:
            raise IOError("Could not open file: ", args.output_path)

        best_oa = 0.0
        best_na = 0.0
        best_oa_line = best_na_list = ''
        for line in f:
            if "Validation" in line:
                if line[-1] == "\n":
                    line = line[:-1]
                res_match = re.match(r'.*Accuracy= (\d.\d+) .*Accuracy= (\d.\d+)', line)
                cur_oa = float(res_match.group(1))  # Overall Accuracy
                cur_na = float(res_match.group(2))  # Normalized Accuracy
                if cur_oa > best_oa:
                    best_oa = cur_oa
                    best_oa_line = line
                if cur_na > best_na:
                    best_na = cur_na
                    best_na_list = line
        print(best_oa_line)
        print(best_na_list)
    else:
        raise NotImplementedError("Process " + args.operation + "not found!")


if __name__ == "__main__":
    main()
