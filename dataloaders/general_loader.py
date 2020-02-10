import os
import random

import scipy
import numpy as np
import imageio

from PIL import Image
from skimage import img_as_float

NUM_CLASSES = 2


def select_super_batch_instances(class_distribution, batch_size=100, super_batch=500):
    instances = []
    overall_count = 0

    samples_per_class = int((batch_size * super_batch) / len(class_distribution))
    # print samples_per_class

    # for each class
    for i in range(len(class_distribution)):
        # print len(class_distribution[i]), samples_per_class
        # (samples_per_class if len(class_distribution[i]) >= samples_per_class else len(class_distribution[i]))
        shuffle = np.asarray(random.sample(range(len(class_distribution[i])), (
            samples_per_class if len(class_distribution[i]) >= samples_per_class else len(class_distribution[i]))))

        for j in shuffle:
            cur_map = class_distribution[i][j][0]
            cur_x = class_distribution[i][j][1]
            cur_y = class_distribution[i][j][2]
            cur_rot = class_distribution[i][j][3]

            instances.append((cur_map, cur_x, cur_y, cur_rot))
            overall_count += 1

    # remaining if int((batch_size*superEpoch)/len(class_distribution)) is not divisible
    # print overall_count, (batch_size*super_batch), overall_count != (batch_size*super_batch)
    if overall_count != (batch_size * super_batch):
        lack = (batch_size * super_batch) - overall_count
        # print 'in', lack
        for i in range(lack):
            rand_class = np.random.randint(len(class_distribution))
            rand_map = np.random.randint(len(class_distribution[rand_class]))

            cur_map = class_distribution[rand_class][rand_map][0]
            cur_x = class_distribution[rand_class][rand_map][1]
            cur_y = class_distribution[rand_class][rand_map][2]
            cur_rot = class_distribution[i][j][3]

            instances.append((cur_map, cur_x, cur_y, cur_rot))
            overall_count += 1

    # print overall_count, (batch_size*super_batch), overall_count != (batch_size*super_batch)
    assert overall_count == (batch_size * super_batch), "Could not select ALL instances"
    # for i in range(len(instances)):
    # print 'Instances ' + str(i)  + ' has length ' + str(len(instances[i]))
    return np.asarray(instances)  # [0]+instances[1]+instances[2]+instances[3]+instances[4]+instances[5]


def select_batch(shuffle, batch_size, it, total_size):
    batch = shuffle[it:min(it + batch_size, total_size)]
    if min(it + batch_size, total_size) == total_size or total_size == it + batch_size:
        shuffle = np.asarray(random.sample(range(total_size), total_size))
        # print "in", shuffle
        it = 0
        if len(batch) < batch_size:
            diff = batch_size - len(batch)
            batch_c = shuffle[it:it + diff]
            batch = np.concatenate((batch, batch_c))
            it = diff
            # print 'c', batch_c, batch, it
    else:
        it += batch_size
    return shuffle, batch, it


def normalize_images(data, mean_full, std_full):
    data[:, :, :, 0] = np.subtract(data[:, :, :, 0], mean_full[0])
    data[:, :, :, 1] = np.subtract(data[:, :, :, 1], mean_full[1])
    data[:, :, :, 2] = np.subtract(data[:, :, :, 2], mean_full[2])

    data[:, :, :, 0] = np.divide(data[:, :, :, 0], std_full[0])
    data[:, :, :, 1] = np.divide(data[:, :, :, 1], std_full[1])
    data[:, :, :, 2] = np.divide(data[:, :, :, 2], std_full[2])


def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0, :]

    return mean_full, std_full


def load_images(path):
    images = []
    masks = []
    names = []

    for f in os.listdir(os.path.join(path, 'JPEGImages')):
        image = img_as_float(imageio.imread(os.path.join(path, 'JPEGImages', f)))
        mask = imageio.imread(os.path.join(path, 'Masks', f[:-4] + "_mask.png"))

        images.append(image)
        masks.append(mask)
        names.append(f[:-4])

    return np.asarray(images), np.asarray(masks), names


def dynamically_create_patches(data, mask_data, training_instances_batch, crop_size, is_train=True):
    patches = []
    classes = NUM_CLASSES * [0]
    classes_patches = []
    masks = []

    overall_count = 0
    flip_count = 0

    for i in range(len(training_instances_batch)):
        cur_map = training_instances_batch[i][0]
        cur_x = training_instances_batch[i][1]
        cur_y = training_instances_batch[i][2]

        cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        if len(cur_patch) != crop_size and len(cur_patch[0]) != crop_size:
            cur_x = cur_x - (crop_size - len(cur_patch))
            cur_y = cur_y - (crop_size - len(cur_patch[0]))
            cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        elif len(cur_patch) != crop_size:
            cur_x = cur_x - (crop_size - len(cur_patch))
            cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        elif len(cur_patch[0]) != crop_size:
            cur_y = cur_y - (crop_size - len(cur_patch[0]))
            cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

        cur_mask_patch = mask_data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

        assert len(cur_patch) == crop_size and len(cur_patch[0]) == crop_size, \
            "Error: Current PATCH size is " + str(len(cur_patch)) + "x" + str(len(cur_patch[0]))

        assert len(cur_mask_patch) == crop_size and len(cur_mask_patch[0]) == crop_size, \
            "Error: Current MASK size is " + str(len(cur_mask_patch)) + "x" + str(len(cur_mask_patch[0]))

        cur_class = np.argmax(np.bincount(cur_mask_patch.astype(int).flatten()))
        classes[int(cur_class)] += 1

        cur_mask = np.ones((crop_size, crop_size), dtype=np.bool)

        # DATA AUGMENTATION
        if is_train is True:
            # ROTATION AUGMENTATION
            cur_rot = training_instances_batch[i][3]
            possible_rotation = np.random.randint(0, 2)
            if possible_rotation == 1:  # default = 1
                # print 'rotation'
                cur_patch = scipy.ndimage.rotate(cur_patch, cur_rot, order=0, reshape=False)
                cur_mask_patch = scipy.ndimage.rotate(cur_mask_patch, cur_rot, order=0, reshape=False)
                cur_mask = scipy.ndimage.rotate(cur_mask, cur_rot, order=0, reshape=False)

            # NORMAL NOISE
            possible_noise = np.random.randint(0, 2)
            if possible_noise == 1:
                cur_patch = cur_patch + np.random.normal(0, 0.01, cur_patch.shape)

            # FLIP AUGMENTATION
            possible_noise = np.random.randint(0, 3)
            if possible_noise == 0:
                patches.append(cur_patch)
                classes_patches.append(cur_mask_patch)
                masks.append(cur_mask)
            if possible_noise == 1:
                patches.append(np.flipud(cur_patch))
                classes_patches.append(np.flipud(cur_mask_patch))
                masks.append(np.flipud(cur_mask))
                flip_count += 1
            elif possible_noise == 2:
                patches.append(np.fliplr(cur_patch))
                classes_patches.append(np.fliplr(cur_mask_patch))
                masks.append(np.fliplr(cur_mask))
                flip_count += 1
        else:
            patches.append(cur_patch)
            classes_patches.append(cur_mask_patch)
            masks.append(cur_mask)

        overall_count += 1

    pt_arr = np.asarray(patches)
    cl_arr = np.asarray(classes_patches, dtype=np.int)
    mk_arr = np.asarray(masks, dtype=np.bool)

    # print pt_arr.shape
    # print cl_arr.shape
    # print mk_arr.shape

    return pt_arr, cl_arr, mk_arr


def create_patches_per_map(data, mask_data, crop_size, stride_crop, index, batch_size):
    patches = []
    classes = []
    pos = []

    h, w, c = data.shape
    # h_m, w_m = mask_data.shape
    total_index_h = (int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
        ((h - crop_size) / stride_crop)) + 2)
    total_index_w = (int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
        ((w - crop_size) / stride_crop)) + 2)

    count = 0

    offset_h = int((index * batch_size) / total_index_w) * stride_crop
    offset_w = int((index * batch_size) % total_index_w) * stride_crop
    first = True

    for j in range(offset_h, total_index_h * stride_crop, stride_crop):
        if first is False:
            offset_w = 0
        for k in range(offset_w, total_index_w * stride_crop, stride_crop):
            if first is True:
                first = False
            cur_x = j
            cur_y = k
            # print j, total_index_h, total_index_h*stride_crop, h
            # print k, total_index_w, total_index_w*stride_crop, w

            # print cur_x, cur_y, cur_x+crop_size, cur_y+crop_size
            # data[cur_x:cur_x+crop_size, cur_y:cur_y+crop_size,:].shape
            patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

            if len(patch) != crop_size and len(patch[0]) != crop_size:
                cur_x = cur_x - (crop_size - len(patch))
                cur_y = cur_y - (crop_size - len(patch[0]))
                patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(patch) != crop_size:
                cur_x = cur_x - (crop_size - len(patch))
                patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(patch[0]) != crop_size:
                cur_y = cur_y - (crop_size - len(patch[0]))
                patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

            count += 1
            cur_mask_patch = mask_data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

            if len(patch) != crop_size or len(patch[0]) != crop_size:
                print("Error: Current patch size ", len(patch), len(patch[0]))
                return
            if len(cur_mask_patch) != crop_size or len(cur_mask_patch[0]) != crop_size:
                print("Error: Current cur_mask_patch size ", len(cur_mask_patch), len(cur_mask_patch[0]))
                return

            patches.append(patch)
            classes.append(cur_mask_patch)
            # print cur_x, cur_y, cur_x+crop_size, cur_y+crop_size, patch.shape, cur_mask_patch.shape

            current_pos = np.zeros(2)
            current_pos[0] = int(cur_x)
            current_pos[1] = int(cur_y)
            pos.append(current_pos)

            if count == batch_size:  # when completes current batch
                # print "--------- batch complete"
                return np.asarray(patches), np.asarray(classes, dtype=np.int8), pos

    # when its not the total size of the batch
    # print "--------- end without batch complete"
    return np.asarray(patches), np.asarray(classes, dtype=np.int8), pos


def create_distributions_over_classes(labels, crop_size, stride_crop):
    classes = [[[] for i in range(0)] for i in range(NUM_CLASSES)]

    for k in range(len(labels)):
        # print labels[k].shape
        w, h = labels[k].shape

        for i in range(0, w, stride_crop):
            for j in range(0, h, stride_crop):
                cur_map = k
                cur_x = i
                cur_y = j
                cur_rot = np.random.randint(0, 360)
                patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if patch_class.shape == (crop_size, crop_size):
                    count = np.bincount(patch_class.astype(int).flatten())
                    classes[int(np.argmax(count))].append((cur_map, cur_x, cur_y, cur_rot))
                else:
                    raise NotImplementedError("Error create_distributions_over_classes: Current patch size is " +
                                              str(len(patch_class)) + "x" + str(len(patch_class[0])))

    for i in range(len(classes)):
        print('Class ' + str(i + 1) + ' has length ' + str(len(classes[i])))

    return classes


def create_prediction_map(img_name, prob_img, size_tuple):
    im_array = np.empty([size_tuple[0], size_tuple[1]], dtype=np.uint8)

    for i in range(size_tuple[0]):
        for j in range(size_tuple[1]):
            im_array[i, j] = int(prob_img[i][j])

    imageio.imwrite(img_name + '.png', im_array * 255)
    img = Image.fromarray(im_array)
    img.save(img_name + "_visual_pred.tif")


def calc_accuracy_by_crop(true_crop, pred_crop, track_conf_matrix, masks=None):
    b, h, w = pred_crop.shape

    acc = 0
    local_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    # count = 0
    for i in range(b):
        for j in range(h):
            for k in range(w):
                if masks is None or (masks is not None and masks[i, j, k]):
                    # count += 1
                    if true_crop[i, j, k] == pred_crop[i, j, k]:
                        acc = acc + 1
                    track_conf_matrix[true_crop[i, j, k]][pred_crop[i, j, k]] += 1
                    local_conf_matrix[true_crop[i, j, k]][pred_crop[i, j, k]] += 1

    # print count, b*h*w
    return acc, local_conf_matrix


def select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, is_loss_or_acc='acc',
                           patch_chosen_values=None, debug=False):
    # if 0 in patch_occur:
    patch_occur[np.where(patch_occur == 0)] = 1
    patch_mean = patch_acc_loss / patch_occur
    # print is_loss_or_acc

    if is_loss_or_acc == 'acc':
        argmax_acc = np.argmax(patch_mean)
        if distribution_type == 'multi_fixed':
            cur_patch_val = int(values[argmax_acc])
        elif distribution_type == 'uniform' or distribution_type == 'multinomial':
            cur_patch_val = values[0] + argmax_acc

        if patch_chosen_values is not None:
            patch_chosen_values[int(argmax_acc)] += 1

        if debug is True:
            print('patch_acc_loss', patch_acc_loss)
            print('patch_occur', patch_occur)
            print('patch_mean', patch_mean)
            print('argmax_acc', argmax_acc)

            print('specific', argmax_acc, patch_acc_loss[argmax_acc], patch_occur[argmax_acc], patch_mean[argmax_acc])

    elif is_loss_or_acc == 'loss':
        arg_sort_out = np.argsort(patch_mean)

        if debug is True:
            print('patch_acc_loss', patch_acc_loss)
            print('patch_occur', patch_occur)
            print('patch_mean', patch_mean)
            print('arg_sort_out', arg_sort_out)
        if distribution_type == 'multi_fixed':
            for i in range(len(values)):
                if patch_occur[arg_sort_out[i]] > 0:
                    cur_patch_val = int(values[arg_sort_out[i]])  # -1*(i+1)
                    if patch_chosen_values is not None:
                        patch_chosen_values[arg_sort_out[i]] += 1
                    if debug is True:
                        print('specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], patch_occur[
                            arg_sort_out[i]], patch_mean[arg_sort_out[i]])
                    break
        elif distribution_type == 'uniform' or distribution_type == 'multinomial':
            for i in range(values[-1] - values[0] + 1):
                if patch_occur[arg_sort_out[i]] > 0:
                    cur_patch_val = values[0] + arg_sort_out[i]
                    if patch_chosen_values is not None:
                        patch_chosen_values[arg_sort_out[i]] += 1
                    if debug is True:
                        print('specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], patch_occur[
                            arg_sort_out[i]], patch_mean[arg_sort_out[i]])
                    break

    if debug is True:
        print('Current patch size ', cur_patch_val)
        if patch_chosen_values is not None:
            print('Distr of chosen sizes ', patch_chosen_values)

    return cur_patch_val
