import numpy as np

def split_train_test(data_distribution, limit=5050):
    train_distrib = []
    test_distrib = []
    for el in data_distribution[1]:
        if el[0] > limit:
            test_distrib.append(el)
        else:
            train_distrib.append(el)

    return np.asarray(train_distrib), np.asarray(test_distrib)
