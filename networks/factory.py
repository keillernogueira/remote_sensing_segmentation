from networks.dynamic_dilated import *


def model_factory(model_name, x, is_training, weight_decay, crop, num_input_bands, num_classes):
    if model_name == 'dilated_grsl':
        logits = dilated_grsl(x, is_training, weight_decay, crop, num_input_bands, num_classes)
    elif model_name == 'dilated_icpr_rate6':
        logits = dilated_icpr_rate6(x, is_training, weight_decay, crop, num_input_bands, num_classes)
    elif model_name == 'dilated_icpr_rate6_densely':
        logits = dilated_icpr_rate6_densely(x, is_training, weight_decay, crop, num_input_bands, num_classes)
    elif model_name == 'dilated_grsl_rate8':
        logits = dilated_grsl_rate8(x, is_training, weight_decay, crop, num_input_bands, num_classes)
    else:
        raise NotImplementedError('Network not identified: ' + model_name)
    return logits
