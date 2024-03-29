import os
import random
import numpy as np
import imageio
import argparse
import matplotlib.pyplot as plt


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(sess, output_path, distribution_type, patch_acc_loss, patch_occur, patch_chosen_values,
                     saver, best_records, step, acc, acc_cls, f1, kappa, iou, cm):
    if len(best_records) < 5:
        best_records.append({'step': step, 'acc': acc, 'acc_cls': acc_cls,
                             'f1': f1, 'kappa': kappa, 'iou': iou, 'cm': cm})

        saver.save(sess, output_path + 'model', global_step=step)
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                or distribution_type == 'multinomial':
            np.save(output_path + 'patch_acc_loss_step_' + str(step) + '.npy', patch_acc_loss)
            np.save(output_path + 'patch_occur_step_' + str(step) + '.npy', patch_occur)
            np.save(output_path + 'patch_chosen_values_step_' + str(step) + '.npy', patch_chosen_values)
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['acc_cls'] > best_records[i]['acc_cls']:
                min_index = i
        # check if currect acc is greater than min saved acc
        if acc_cls > best_records[min_index]['acc_cls']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['step'])
            os.remove(os.path.join(output_path, 'model-' + min_step + '.index'))
            os.remove(os.path.join(output_path, 'model-' + min_step + '.data-00000-of-00001'))
            os.remove(os.path.join(output_path, 'model-' + min_step + '.meta'))
            if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                    or distribution_type == 'multinomial':
                os.remove(os.path.join(output_path, 'patch_acc_loss_step_' + min_step + '.npy'))
                os.remove(os.path.join(output_path, 'patch_occur_step_' + min_step + '.npy'))
                os.remove(os.path.join(output_path, 'patch_chosen_values_step_' + min_step + '.npy'))

            # replace min value with current
            best_records[min_index] = {'step': step, 'acc': acc, 'acc_cls': acc_cls,
                                       'f1': f1, 'kappa': kappa, 'iou': iou, 'cm': cm}
            # save current model
            saver.save(sess, output_path + 'model', global_step=step)
            if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                    or distribution_type == 'multinomial':
                np.save(output_path + 'patch_acc_loss_step_' + str(step) + '.npy', patch_acc_loss)
                np.save(output_path + 'patch_occur_step_' + str(step) + '.npy', patch_occur)
                np.save(output_path + 'patch_chosen_values_step_' + str(step) + '.npy', patch_chosen_values)


def define_multinomial_probs(values, dif_prob=2):
    interval_size = values[-1] - values[0] + 1

    general_prob = 1.0 / float(interval_size)
    max_prob = general_prob * dif_prob  # for values

    probs = np.full(interval_size, (1.0 - max_prob * len(values)) / float(interval_size - len(values)))
    for i in range(len(values)):
        probs[values[i] - values[0]] = max_prob

    return probs


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


def create_prediction_map(img_name, prob_img, channels=False):
    if channels is True:
        for i in range(16):  # range(prob_img.shape[-1]):
            # imageio.imwrite(img_name + 'feat_' + str(i) + '.png', prob_img[:, :, i].astype(np.uint8))
            plt.imsave(img_name + 'feat_' + str(i) + '.png', prob_img[:, :, i], cmap=plt.cm.jet)
    else:
        imageio.imwrite(img_name + '.png', prob_img.astype(np.uint8) * 255)
        # img = Image.fromarray(prob_img.astype(np.uint8) * 255)
        # img.save(img_name + ".tif")


def calc_accuracy_by_crop(true_crop, pred_crop, num_classes, track_conf_matrix, masks=None):
    b, h, w = pred_crop.shape

    acc = 0
    local_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
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


def calc_accuracy_by_class(true_crop, pred_crop, num_classes, track_conf_matrix):
    acc = 0
    local_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
    # count = 0
    for i in range(len(true_crop)):
        if true_crop[i] == pred_crop[i]:
            acc = acc + 1
        track_conf_matrix[true_crop[i]][pred_crop[i]] += 1
        local_conf_matrix[true_crop[i]][pred_crop[i]] += 1

    return acc, local_conf_matrix


def create_cm(true, pred):
    conf_matrix = np.zeros((len(np.unique(true)), len(np.unique(true))), dtype=np.uint32)
    c, h, w = true.shape
    for i in range(c):
        for j in range(h):
            for k in range(w):
                conf_matrix[true[i, j, k]][pred[i, j, k]] += 1

    return conf_matrix


def kappa_with_cm(conf_matrix):
    acc = 0
    marginal = 0
    total = float(np.sum(conf_matrix))
    for i in range(len(conf_matrix)):
        acc += conf_matrix[i][i]
        marginal += np.sum(conf_matrix, 0)[i] * np.sum(conf_matrix, 1)[i]

    kappa = (total * acc - marginal) / (total * total - marginal)
    return kappa


def f1_with_cm(conf_matrix):
    precision = [0] * len(conf_matrix)
    recall = [0] * len(conf_matrix)
    f1 = [0] * len(conf_matrix)
    for i in range(len(conf_matrix)):
        precision[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 0)[i])
        recall[i] = conf_matrix[i][i] / float(np.sum(conf_matrix, 1)[i])
        f1[i] = 2 * ((precision[i]*recall[i])/(precision[i]+recall[i]))

    return np.mean(f1)


def generate_gt_road(input_path, output_path):
    img = imageio.imread(input_path)
    print(img.shape)
    from skimage.morphology import dilation, disk
    mask = dilation(img, disk(3))
    imageio.imwrite(output_path, mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='utils')
    # general options
    parser.add_argument('--input_image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--output_image_path', type=str, required=True, help='Output image path')
    args = parser.parse_args()
    print(args)
    img = imageio.imread(args.input_image_path)
    print(img.shape)
    for i in range(img.shape[2]):
        print(i+1, np.min(img[:, :, i]), np.max(img[:, :, i]))
    print(np.count_nonzero(np.isnan(img)))
    print(img[500:550, 500:550, :])
    # generate_gt_road(args.input_image_path, args.output_image_path)
