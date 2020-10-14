import os
import numpy as np
from sklearn.model_selection import KFold


def main():
    folder = 'C:\\Users\\keill\\Desktop\\Datasets\\Bacia_SantoAntonio_Revisado\\'

    names = []
    for f in os.listdir(os.path.join(folder, 'Raster_Original')):
        names.append(f)
    names = np.asarray(names)

    kf = KFold(n_splits=5)

    fold = 1
    for train_index, test_index in kf.split(names):
        train = names[train_index]
        test = names[test_index]
        print(train.shape, test.shape)
        np.savetxt(os.path.join(folder, 'train_fold_' + str(fold) + '.txt'), train, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(folder, 'test_fold_' + str(fold) + '.txt'), test, fmt='%s', delimiter=',')
        fold += 1


if __name__ == "__main__":
    main()
