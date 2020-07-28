from networks.dynamic_dilated import *
from networks.fcn import *
from networks.pixelwise import *
from networks.segnet import *
from networks.unet import *
from networks.deeplab import deeplab


def model_factory(model_name, x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size):
    if model_name == 'dilated_grsl':
        logits = dilated_grsl(x, is_training, weight_decay, crop, num_input_bands, num_classes)
    elif model_name == 'dilated_icpr_rate6':
        logits = dilated_icpr_rate6(x, is_training, weight_decay, crop, num_input_bands, num_classes)
    elif model_name == 'dilated_icpr_rate6_densely':
        logits = dilated_icpr_rate6_densely(x, is_training, weight_decay, crop, num_input_bands, num_classes)
    elif model_name == 'dilated_grsl_rate8':
        logits = dilated_grsl_rate8(x, is_training, weight_decay, crop, num_input_bands, num_classes)

    elif model_name == 'fcn_25_1_4x':
        logits = fcn_25_1_4x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes)
    elif model_name == 'fcn_25_2_2x':
        logits = fcn_25_2_2x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes)
    elif model_name == 'fcn_25_3_2x_icpr':
        logits = fcn_25_3_2x_icpr(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes)
    elif model_name == 'fcn_50_1_8x':
        logits = fcn_50_1_8x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes)
    elif model_name == 'fcn_50_2_4x':
        logits = fcn_50_2_4x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes)
    elif model_name == 'fcn_50_3_2x':
        logits = fcn_50_3_2x(x, dropout, is_training, crop, weight_decay, num_input_bands, num_classes)

    elif model_name == 'pixelwise':
        logits = pixelwise(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes)

    elif model_name == 'segnet':
        logits = segnet(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size)
    elif model_name == 'segnet_4':
        logits = segnet_4(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size)

    elif model_name == 'unet':
        logits = unet(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size)
    elif model_name == 'unet_4':
        logits = unet_4(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size)

    elif model_name == 'deeplabv3+':
        logits = deeplab(x, dropout, is_training, weight_decay, crop, num_input_bands, num_classes, crop_size)
    else:
        raise NotImplementedError('Network not identified: ' + model_name)
    return logits
