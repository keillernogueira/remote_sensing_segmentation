import os

import numpy as np
import imageio

from skimage import img_as_float
from skimage.morphology import dilation, disk

from dataloaders.utils import create_or_load_mean, create_distrib_multi_images


class TrainValTestLoader:

    def __init__(self, dataset, dataset_input_path, num_classes, output_path,
                 model, reference_crop_size, reference_stride_crop,
                 is_validation, simulate_images=False):
        super().__init__()

        self.dataset = dataset
        self.dataset_input_path = dataset_input_path

        self.num_classes = num_classes
        self.output_path = output_path

        self.reference_crop_size = reference_crop_size
        self.reference_stride_crop = reference_stride_crop

        self._mean = None
        self._std = None

        self.train_distrib = None
        self.test_distrib = None

        if dataset == 'tree':
            self.train_data, self.train_labels, _ = self.load_images('train', simulate_images=simulate_images)
            self.test_data, self.test_labels, self.test_name = self.load_images(
                ('validation' if is_validation else 'test'), simulate_images=simulate_images)
        elif dataset == 'road_detection':
            self.train_data, self.train_labels, _ = self.load_images_road(['2', '3'], simulate_images=simulate_images)
            self.test_data, self.test_labels, self.test_name = self.load_images_road(['4'],
                                                                                     simulate_images=simulate_images)
        elif dataset == 'river':
            self.train_data, self.train_labels, _ = self.load_images_river('train', fold=1)
            self.test_data, self.test_labels, self.test_name = self.load_images_river('test', fold=1)
        else:
            raise NotImplementedError('DataLoader not identified: ' + dataset)

        print(self.train_data.shape, self.train_labels.shape)
        print(self.test_data.shape, self.test_labels.shape)
        self.train_distrib = create_distrib_multi_images(self.train_labels, model, self.reference_crop_size,
                                                         self.reference_stride_crop, self.num_classes,
                                                         filtering_non_classes=(dataset == 'road_detection' or
                                                                                dataset == 'river'),
                                                         percentage_filter=(0.8 if dataset == 'road_detection' else 0.99),
                                                         percentage_pos_class=(0.1 if dataset == 'road_detection' else 0.5))
        # using reference_crop_size instead of reference_stride_crop for the line BELOW
        # to allow a better validation without intersection of the patches
        self.test_distrib = create_distrib_multi_images(self.test_labels, model, self.reference_crop_size,
                                                        self.reference_crop_size, self.num_classes, self.dataset)
        print(len(self.train_distrib), len(self.test_distrib))
        self._mean, self._std = create_or_load_mean(self.train_data, self.train_distrib,
                                                    self.reference_crop_size, self.reference_stride_crop,
                                                    self.output_path)

    def load_images(self, stage, simulate_images=False):
        print(stage)
        images = []
        masks = []
        names = []

        if simulate_images is True:
            mask = np.random.rand(580, 256, 256, 3)
            images = np.random.rand(580, 256, 256, 3)
            return np.asarray(images), np.asarray(mask)

        for f in os.listdir(os.path.join(self.dataset_input_path, stage, 'images')):
            names.append(f)

            # print(os.path.join(self.dataset_input_path, stage, 'images', f))
            image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, stage, 'images', f)))
            images.append(image)

            # print(os.path.join(self.dataset_input_path, stage, 'labels', f))
            mask = (imageio.imread(os.path.join(self.dataset_input_path, stage, 'labels', f))/255).astype(int)
            masks.append(mask)

        return np.asarray(images), np.asarray(masks), np.asarray(names)

    def load_images_road(self, areas, simulate_images=False):
        images = []
        masks = []
        names = []

        if simulate_images is True:
            mask = np.random.rand(580, 256, 256, 3)
            images = np.random.rand(580, 256, 256, 3)
            return np.asarray(images), np.asarray(mask)

        for i, area in enumerate(areas):
            names.append('area' + area)

            # print(os.path.join(self.dataset_input_path, stage, 'images', f))
            image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, 'area' + area +
                                                             '_landsat8_toa_2013_pansharpen.tif')))
            image[np.where(np.isnan(image))] = 0  # replace nan values with 0's
            images.append(image)

            # print(os.path.join(self.dataset_input_path, stage, 'labels', f))
            mask = (imageio.imread(os.path.join(self.dataset_input_path, 'area' + area + '_mask.png'))).astype(int)
            masks.append(mask)

        return np.asarray(images), np.asarray(masks), np.asarray(names)

    def load_images_river(self, stage, fold=1):
        images = []
        masks = []
        names = []

        for f in np.loadtxt(os.path.join(self.dataset_input_path, stage + '_fold_' + str(fold) + '.txt'), dtype='str'):
            names.append(f)

            # print(os.path.join(self.dataset_input_path, stage, 'images', f))
            image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, 'Raster_Original',
                                                             '8Bit_Raster_Foto_' + f.split(".")[0][-1] + '.tif')))
            images.append(image[0:512, 0:512])

            # print(os.path.join(self.dataset_input_path, stage, 'labels', f))
            mask = imageio.imread(os.path.join(self.dataset_input_path, 'Raster_Rotulado', f)).astype(int)
            mask[np.where(mask == 3)] = 1  # change pixels with values 3 to 1
            assert(len(np.bincount(mask.flatten())) == 2)
            masks.append(mask[0:512, 0:512])

        return np.asarray(images), np.asarray(masks), np.asarray(names)
