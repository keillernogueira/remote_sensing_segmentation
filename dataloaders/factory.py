from dataloaders.unique_image_loader import UniqueImageLoader
from dataloaders.train_validation_test_loader import TrainValTestLoader


def dataloader_factory(dataset, dataset_input_path, dataset_gt_path, num_classes, output_path,
                       model_name, reference_crop_size, reference_stride_crop,
                       is_validation, simulate_dataset):
    if dataset == 'laranja':
        return UniqueImageLoader(dataset, dataset_input_path, dataset_gt_path, num_classes, output_path,
                                 model_name, reference_crop_size, reference_stride_crop,
                                 simulate_dataset)
    elif dataset == 'arvore' or dataset == 'road_detection':
        return TrainValTestLoader(dataset, dataset_input_path, num_classes, output_path,
                                  model_name, reference_crop_size, reference_stride_crop,
                                  is_validation, simulate_dataset)
    else:
        raise NotImplementedError('DataLoader not identified: ' + dataset)
