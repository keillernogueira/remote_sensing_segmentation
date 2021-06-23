import numpy as np
import imageio
from skimage import img_as_float


def save_image(img, cur_x, cur_y, crop_size):
    output = np.zeros((img.shape[0], img.shape[1], 3))
    output[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :] = 255
    imageio.imwrite('output.png', output.astype(np.uint8))


def statistics_specific_band(cur_x, cur_y, crop_size):
    print("--------------------------------------------------------------statistics_specific_band")
    img = "/home/kno/dataset_laranjal/Dataset_Laranjal/Parrot_Sequoia/all/Sequoia_Band03RedEdge_v3.tif"

    print(img)
    image_orig = imageio.imread(img)
    crop_orig = image_orig[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
    _mean = np.mean(np.mean(crop_orig, axis=0), axis=0)
    print(np.min(crop_orig), np.max(crop_orig), _mean)
    print(crop_orig)

    print('1', cur_x + 0, cur_y + 31, image_orig[cur_x + 0, cur_y + 31])
    print('2', cur_x + 1, cur_y + 31, image_orig[cur_x + 1, cur_y + 31])
    print('3', cur_x + 2, cur_y + 31, image_orig[cur_x + 2, cur_y + 31])

    # image_float = img_as_float(image_orig)
    # crop_float = image_float[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
    # _mean = np.mean(np.mean(crop_float, axis=0), axis=0)
    # print(np.min(crop_float), np.max(crop_float), _mean)
    # print(crop_float)

    counter = 0
    # pos = []
    h, w = crop_orig.shape
    for i in range(h):
        for j in range(w):
            if crop_orig[i, j] < 0:
                # if np.random.rand(1, 1)[0] < 0.1:  # -3.402823e+38
                print(i, j, image_orig[i, j])
                # pos.append((i, j))
                counter += 1
    print(counter)
    # print(pos)

    return image_orig


def main():
    images = ["/home/kno/dataset_laranjal/Dataset_Laranjal/Parrot_Sequoia/all/Sequoia_Band01Green_v3.tif",
              "/home/kno/dataset_laranjal/Dataset_Laranjal/Parrot_Sequoia/all/Sequoia_Band02Red_v3.tif",
              "/home/kno/dataset_laranjal/Dataset_Laranjal/Parrot_Sequoia/all/Sequoia_Band03RedEdge_v3.tif",
              "/home/kno/dataset_laranjal/Dataset_Laranjal/Parrot_Sequoia/all/Sequoia_Band04NIR_v3.tif"]

    cur_x = 1200
    cur_y = 7600
    crop_size = 32

    # for img in images:
    #     print(img)
    #     image_orig = imageio.imread(img)
    #     print('1', cur_x + 0, cur_y + 31, image_orig[cur_x + 0, cur_y + 31])
    #     print('2', cur_x + 1, cur_y + 31, image_orig[cur_x + 1, cur_y + 31])
    #     print('3', cur_x + 2, cur_y + 31, image_orig[cur_x + 2, cur_y + 31])
    #     crop_orig = image_orig[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
    #     _mean = np.mean(np.mean(crop_orig, axis=0), axis=0)
    #     print(np.min(crop_orig), np.max(crop_orig), _mean)
    #     print(crop_orig)

    band = statistics_specific_band(cur_x, cur_y, crop_size)

    save_image(band, cur_x, cur_y, crop_size)


if __name__ == "__main__":
    main()
