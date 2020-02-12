import os

import numpy as np
import imageio

from skimage import img_as_float

from dataloaders.general_loader import GeneralLoader
from dataloaders.spliters import split_train_test


class UniqueImageLoader(GeneralLoader):

    def __init__(self, dataset_input_path, dataset_gt_path, num_classes, output_path):
        super().__init__()

        self.dataset_input_path = dataset_input_path
        self.dataset_gt_path = dataset_gt_path
        self.num_classes = num_classes
        self.output_path = output_path

        self.data, self.labels = self.load_images()

    def load_images(self, concatenate_images_in_depth=True):
        first = True
        images = []
        for f in os.listdir(self.dataset_input_path):
            # break
            image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, f)))

            if concatenate_images_in_depth is True:
                if first is True:
                    images = np.expand_dims(image, axis=2)
                    first = False
                else:
                    images = np.concatenate((images, np.expand_dims(image, axis=2)), axis=2)
            else:
                images.append(image)

        # images = np.random.rand(8777, 12148, 4)
        mask = imageio.imread(self.dataset_gt_path)

        return np.asarray(images), np.asarray(mask)

    def split_dataset(self, crop_size, stride_crop, dataset_split_method):
        data_distribution = self.create_distributions_over_classes(crop_size, stride_crop)

        # splitting dataset
        if dataset_split_method == 'train_test':
            if os.path.isfile(os.path.join(self.output_path, 'train_distrib.npy')):
                train_distrib = np.load(os.path.join(self.output_path, 'train_distrib.npy'), allow_pickle=True)
                test_distrib = np.load(os.path.join(self.output_path, 'test_distrib.npy'), allow_pickle=True)
            else:
                train_distrib, test_distrib = split_train_test(data_distribution)
                np.save(os.path.join(self.output_path, 'train_distrib.npy'), train_distrib)
                np.save(os.path.join(self.output_path, 'test_distrib.npy'), test_distrib)
        else:
            raise NotImplementedError("Dataset split method " + dataset_split_method + " not implemented.")

        self.train_distrib = train_distrib
        self.test_distrib = test_distrib
