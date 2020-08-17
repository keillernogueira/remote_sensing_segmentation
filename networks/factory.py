from networks.dynamic_dilated import *
from networks.fcn import *
from networks.pixelwise import *
from networks.segnet import *
from networks.unet import *
from networks.deeplab import deeplab


def model_factory(model_name, x, dropout, is_training, weight_decay,
                  crop, num_input_bands, num_classes, crop_size, extract_features=False):
    if model_name == 'dilated_grsl':
        return dilated_grsl(x, is_training, weight_decay, crop, num_input_bands, num_classes, extract_features)
    elif model_name == 'dilated_icpr_rate6':
        return dilated_icpr_rate6(x, is_training, weight_decay, crop, num_input_bands, num_classes, extract_features)
    elif model_name == 'dilated_icpr_rate6_densely':
        return dilated_icpr_rate6_densely(x, is_training, weight_decay, crop, num_input_bands, num_classes, extract_features)
    elif model_name == 'dilated_grsl_rate8':
        return dilated_grsl_rate8(x, is_training, weight_decay, crop, num_input_bands, num_classes, extract_features)

    elif model_name == 'fcn_25_1_4x':
        return fcn_25_1_4x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes, extract_features)
    elif model_name == 'fcn_25_2_2x':
        return fcn_25_2_2x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes, extract_features)
    elif model_name == 'fcn_25_3_2x_icpr':
        return fcn_25_3_2x_icpr(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes, extract_features)
    elif model_name == 'fcn_50_1_8x':
        return fcn_50_1_8x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes, extract_features)
    elif model_name == 'fcn_50_2_4x':
        return fcn_50_2_4x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes, extract_features)
    elif model_name == 'fcn_50_3_2x':
        return fcn_50_3_2x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes, extract_features)

    elif model_name == 'pixelwise':
        return pixelwise(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, extract_features)

    elif model_name == 'segnet':
        return segnet(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features)
    elif model_name == 'segnet_4':
        return segnet_4(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features)

    elif model_name == 'unet':
        return unet(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features)
    elif model_name == 'unet_4':
        return unet_4(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features)
    elif model_name == 'unet_road_detection':
        return unet_road_detection(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features)

    elif model_name == 'deeplabv3+':
        return deeplab(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size, extract_features)
    else:
        raise NotImplementedError('Network not identified: ' + model_name)
