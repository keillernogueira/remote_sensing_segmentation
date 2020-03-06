import numpy as np


def split_train_test(model, data_distribution, limit=5050):
    train_distrib = []
    test_distrib = []
    if model == 'pixelwise':
        data = data_distribution[0] + data_distribution[1]
    else:
        data = data_distribution[1]

    for el in data:
        if el[0] > limit:
            test_distrib.append(el)
        else:
            train_distrib.append(el)

    return np.asarray(train_distrib), np.asarray(test_distrib)
