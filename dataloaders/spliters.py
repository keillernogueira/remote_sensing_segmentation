import numpy as np


def split_train_test(model, data_distribution, limit=5050):
    train_distrib = []
    test_distrib = []
    if model == 'pixelwise':
        size_0 = len(data_distribution[0])
        size_1 = len(data_distribution[1])
        ratio = size_1 / float(size_0)
        print(ratio, size_0, size_1)
        count_0 = 0
        count_1 = 0
        for el in data_distribution[0]:
            if el[0] > limit:
                test_distrib.append(el)
            else:
                if np.random.rand(1, 1)[0] < ratio:
                    count_0 += 1
                    train_distrib.append(el)
        for el in data_distribution[1]:
            if el[0] > limit:
                test_distrib.append(el)
            else:
                count_1 += 1
                train_distrib.append(el)
        print(count_0, count_1)
    else:
        for el in data_distribution[1]:
            if el[0] > limit:
                test_distrib.append(el)
            else:
                train_distrib.append(el)

    return np.asarray(train_distrib), np.asarray(test_distrib)
