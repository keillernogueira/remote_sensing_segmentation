from networks.dynamic_dilated import *


def model_factory(model_name, x, is_training, weight_decay, crop):
    # CONVNET
    if model_name == 'dilated_icpr_original':
        logits = dilated_grsl(x, is_training, weight_decay, crop)
    elif model_name == 'dilated_icpr_rate6':
        logits = dilated_icpr_rate6(x, is_training, weight_decay, crop)
    elif model_name == 'dilated_icpr_rate6_densely':
        logits = dilated_icpr_rate6_densely(x, is_training, weight_decay, crop)
    elif model_name == 'dilated8_grsl':
        logits = dilated_grsl_rate8(x, is_training, weight_decay, crop)
    else:
        raise NotImplementedError('Network not identified: ' + model_name)
    return logits
