import os

import numpy as np
import imageio

from skimage import img_as_float

from dataloaders.utils import create_or_load_mean, create_distributions_over_classes
from dataloaders.spliters import split_train_test


class UniqueImageLoader:

    def __init__(self, dataset_input_path, dataset_gt_path, num_classes, output_path,
                 model_name, reference_crop_size, reference_stride_crop,
                 simulate_images=False):
        super().__init__()

        self.dataset_input_path = dataset_input_path
        self.dataset_gt_path = dataset_gt_path
        self.num_classes = num_classes
        self.output_path = output_path

        self.reference_crop_size = reference_crop_size
        self.reference_stride_crop = reference_stride_crop

        self._mean = None
        self._std = None

        self.train_distrib = None
        self.test_distrib = None

        self.data, self.labels = self.load_images(simulate_images=simulate_images)
        print(self.data.shape, self.labels.shape)
        self.train_distrib, self.test_distrib = self.split_dataset(model_name)
        print(len(self.train_distrib), len(self.test_distrib))
        self._mean, self._std = create_or_load_mean(self.data, self.train_distrib,
                                                    reference_crop_size, reference_stride_crop, self.output_path)

    def load_images(self, concatenate_images_in_depth=True, simulate_images=False):
        first = True
        images = []

        if simulate_images is True:
            mask = imageio.imread(self.dataset_gt_path)
            images = np.random.rand(mask.shape[0], mask.shape[1], 4)
            # images = np.random.rand(8777, 12148, 4)
            print(mask.shape, images.shape)
            return np.asarray(images), np.asarray(mask)

        for f in os.listdir(self.dataset_input_path):
            print(f)
            image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, f)))

            if concatenate_images_in_depth is True:
                if first is True:
                    images = np.expand_dims(image, axis=2)
                    first = False
                else:
                    images = np.concatenate((images, np.expand_dims(image, axis=2)), axis=2)
            else:
                images.append(image)

        mask = imageio.imread(self.dataset_gt_path)

        return np.asarray(images), np.asarray(mask)

    def split_dataset(self, model, dataset_split_method='train_test'):
        # splitting dataset
        if dataset_split_method == 'train_test':
            if os.path.isfile(os.path.join(self.output_path, 'train_distrib.npy')):
                train_distrib = np.load(os.path.join(self.output_path, 'train_distrib.npy'), allow_pickle=True)
                test_distrib = np.load(os.path.join(self.output_path, 'test_distrib.npy'), allow_pickle=True)
            else:
                data_distribution = create_distributions_over_classes(self.labels, model, self.reference_crop_size,
                                                                      self.reference_stride_crop, self.num_classes)
                train_distrib, test_distrib = split_train_test(model, data_distribution)
                np.save(os.path.join(self.output_path, 'train_distrib.npy'), train_distrib)
                np.save(os.path.join(self.output_path, 'test_distrib.npy'), test_distrib)
        else:
            raise NotImplementedError("Dataset split method " + dataset_split_method + " not implemented.")

        return train_distrib, test_distrib
