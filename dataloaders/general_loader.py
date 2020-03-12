import abc
import os

import scipy
import numpy as np


class GeneralLoader:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dataset_input_path = None
        self.dataset_gt_path = None
        self.num_classes = None
        self.output_path = None

        self.labels = None
        self.data = None

        self._mean = None
        self._std = None

        self.train_distrib = None
        self.test_distrib = None

        self.train_patches = None
        self.train_labels = None
        self.train_masks = None

        self.test_patches = None
        self.test_labels = None
        self.test_masks = None

    def create_distributions_over_classes(self, model, crop_size, stride_size):
        classes = [[[] for i in range(0)] for i in range(self.num_classes)]

        w, h = self.labels.shape

        for i in range(0, w, stride_size):
            for j in range(0, h, stride_size):
                cur_x = i
                cur_y = j
                patch_class = self.labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = self.labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    patch_class = self.labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = self.labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                assert patch_class.shape == (crop_size, crop_size), "Error create_distributions_over_classes: " \
                                                                    "Current patch size is " + str(len(patch_class)) + \
                                                                    "x" + str(len(patch_class[0]))

                count = np.bincount(patch_class.astype(int).flatten())
                if len(count) == 2:
                    if model != 'pixelwise':
                        classes[1].append((cur_x, cur_y, np.bincount(patch_class.flatten())))
                    else:
                        _cl = self.labels[cur_x + int(crop_size/2) - 1, cur_x + int(crop_size/2) - 1]
                        # print('c', crop_size, cur_x, cur_y, _cl)
                        classes[_cl].append((cur_x, cur_y, np.bincount(patch_class.flatten())))
                else:
                    classes[0].append((cur_x, cur_y, np.bincount(patch_class.flatten())))

        for i in range(len(classes)):
            print('Class ' + str(i + 1) + ' has length ' + str(len(classes[i])))

        return classes

    def normalize_images(self, data):
        for i in range(len(self._mean)):
            data[:, :, :, i] = np.subtract(data[:, :, :, i], self._mean[i])
            data[:, :, :, i] = np.divide(data[:, :, :, i], self._std[i])

        # data[:, :, :, 0] = np.subtract(data[:, :, :, 0], self._mean[0])
        # data[:, :, :, 1] = np.subtract(data[:, :, :, 1], self._mean[1])
        # data[:, :, :, 2] = np.subtract(data[:, :, :, 2], self._mean[2])
        # data[:, :, :, 3] = np.subtract(data[:, :, :, 3], self._mean[3])
        #
        # data[:, :, :, 0] = np.divide(data[:, :, :, 0], self._std[0])
        # data[:, :, :, 1] = np.divide(data[:, :, :, 1], self._std[1])
        # data[:, :, :, 2] = np.divide(data[:, :, :, 2], self._std[2])
        # data[:, :, :, 3] = np.divide(data[:, :, :, 3], self._std[3])

    def compute_image_mean(self, data):
        _mean = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
        _std = np.std(data, axis=0, ddof=1)[0, 0, :]

        return _mean, _std

    def dynamically_calculate_mean_and_std(self, train_distrib, crop_size):
        mean_full = []
        std_full = []

        all_patches = []
        # count = 0
        for i in range(len(train_distrib)):
            cur_x = train_distrib[i][0]
            cur_y = train_distrib[i][1]

            patches = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            if len(patches[0]) != crop_size or len(patches[1]) != crop_size:
                raise SystemError("Error! Current patch size: " + str(len(patches)) + "x" + str(len(patches[0])))

            all_patches.append(patches)

            if i > 0 and i % 5000 == 0:
                mean, std = self.compute_image_mean(np.asarray(all_patches))
                print(np.min(mean), np.max(mean))
                print(np.min(std), np.max(std))
                mean_full.append(mean)
                std_full.append(std)
                all_patches = []

        # remaining images
        mean, std = self.compute_image_mean(np.asarray(all_patches))
        mean_full.append(mean)
        std_full.append(std)

        # print 'count', count
        print(np.min(mean_full), np.max(mean_full))
        print(np.min(std_full), np.max(std_full))
        return np.mean(mean_full, axis=0), np.mean(std_full, axis=0)

    def create_or_load_mean(self, crop_size, stride_size):
        # create mean, std from training
        if os.path.isfile(os.path.join(self.output_path, 'crop_' + str(crop_size) + '_stride_' +
                          str(stride_size) + '_mean.npy')):
            self._mean = np.load(os.path.join(self.output_path, 'crop_' + str(crop_size) + '_stride_' +
                                 str(stride_size) + '_mean.npy'), allow_pickle=True)
            self._std = np.load(os.path.join(self.output_path, 'crop_' + str(crop_size) + '_stride_' +
                                str(stride_size) + '_std.npy'), allow_pickle=True)
        else:
            # try:
            #     print('-----------------------------------------------------------------all patches')
            #     patches, _, _ = self.create_patches(self.train_distrib, crop_size, is_train=False)
            #     self._mean, self._std = self.compute_image_mean(patches)
            # except MemoryError:
            #     print('-----------------------------------------------------------------dynamic patches')
            self._mean, self._std = self.dynamically_calculate_mean_and_std(self.train_distrib, crop_size)
            np.save(os.path.join(self.output_path, 'crop_' + str(crop_size) + '_stride_' +
                    str(stride_size) + '_mean.npy'), self._mean)
            np.save(os.path.join(self.output_path, 'crop_' + str(crop_size) + '_stride_' +
                    str(stride_size) + '_std.npy'), self._std)
        print(self._mean, self._std)

    def create_patches(self, train_instances, crop_size, is_train=True):
        patches = []
        classes = self.num_classes * [0]
        classes_patches = []
        masks = []

        overall_count = 0
        flip_count = 0

        for i in range(len(train_instances)):
            cur_x = train_instances[i][0]
            cur_y = train_instances[i][1]

            cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            if len(cur_patch) != crop_size and len(cur_patch[0]) != crop_size:
                cur_x = cur_x - (crop_size - len(cur_patch))
                cur_y = cur_y - (crop_size - len(cur_patch[0]))
                cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(cur_patch) != crop_size:
                cur_x = cur_x - (crop_size - len(cur_patch))
                cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(cur_patch[0]) != crop_size:
                cur_y = cur_y - (crop_size - len(cur_patch[0]))
                cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

            cur_mask_patch = self.labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

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
                cur_rot = np.random.randint(0, 360)
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
        mask_arr = np.asarray(masks, dtype=np.bool)

        if is_train is True:
            self.train_patches = pt_arr
            self.train_labels = cl_arr
            self.train_masks = mask_arr
        else:
            self.test_patches = pt_arr
            self.test_labels = cl_arr
            self.test_masks = mask_arr

        return pt_arr, cl_arr, mask_arr

    def dynamically_create_patches(self, model, train_instances, crop_size, is_train=True):
        patches = []
        classes = self.num_classes * [0]
        classes_patches = []
        masks = []

        overall_count = 0
        flip_count = 0

        for i in range(len(train_instances)):
            cur_x = train_instances[i][0]
            cur_y = train_instances[i][1]

            cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            # print('before', np.min(cur_patch), np.max(cur_patch), np.isnan(cur_patch).any(), cur_x, cur_y)
            if len(cur_patch) != crop_size and len(cur_patch[0]) != crop_size:
                cur_x = cur_x - (crop_size - len(cur_patch))
                cur_y = cur_y - (crop_size - len(cur_patch[0]))
                cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(cur_patch) != crop_size:
                cur_x = cur_x - (crop_size - len(cur_patch))
                cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(cur_patch[0]) != crop_size:
                cur_y = cur_y - (crop_size - len(cur_patch[0]))
                cur_patch = self.data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            assert len(cur_patch) == crop_size and len(cur_patch[0]) == crop_size, \
                "Error: Current PATCH size is " + str(len(cur_patch)) + "x" + str(len(cur_patch[0]))
            # print('after', np.min(cur_patch), np.max(cur_patch), np.isnan(cur_patch).any(), cur_x, cur_y)

            if model == 'pixelwise':
                cur_mask_patch = self.labels[cur_x + int(crop_size/2) - 1, cur_x + int(crop_size/2) - 1]
            else:
                cur_mask_patch = self.labels[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                assert len(cur_mask_patch) == crop_size and len(cur_mask_patch[0]) == crop_size, \
                    "Error: Current MASK size is " + str(len(cur_mask_patch)) + "x" + str(len(cur_mask_patch[0]))

            # cur_class = np.argmax(np.bincount(cur_mask_patch.astype(int).flatten()))
            # classes[int(cur_class)] += 1

            cur_mask = np.ones((crop_size, crop_size), dtype=np.bool)

            # DATA AUGMENTATION
            if is_train is True:
                # ROTATION AUGMENTATION
                cur_rot = np.random.randint(0, 360)
                possible_rotation = np.random.randint(0, 2)
                if possible_rotation == 1:  # model != 'pixelwise' and
                    # print 'rotation'
                    cur_patch = scipy.ndimage.rotate(cur_patch, cur_rot, order=0, reshape=False)
                    if model == 'pixelwise':
                        cur_mask_patch = cur_mask_patch
                    else:
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
                    if model == 'pixelwise':
                        classes_patches.append(cur_mask_patch)
                    else:
                        classes_patches.append(np.flipud(cur_mask_patch))
                    masks.append(np.flipud(cur_mask))
                    flip_count += 1
                elif possible_noise == 2:
                    patches.append(np.fliplr(cur_patch))
                    if model == 'pixelwise':
                        classes_patches.append(cur_mask_patch)
                    else:
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

        if is_train is True:
            self.train_patches = pt_arr
            self.train_labels = cl_arr
            self.train_masks = mk_arr
        else:
            self.test_patches = pt_arr
            self.test_labels = cl_arr
            self.test_masks = mk_arr

        return pt_arr, cl_arr, mk_arr

    def create_patches_per_map(self, data, mask_data, crop_size, stride_crop, index, batch_size):
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

