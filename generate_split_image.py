import imageio
import numpy as np
import PIL
from PIL import Image


def resize_image(img_path, output_path):
    basewidth = 1000
    img = Image.open(img_path)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(output_path)


def save_train_test_image(img, output_image_path):
    h, w = img.shape
    img_train_test = np.zeros([h, w, 3], dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                if i > 5050:
                    img_train_test[i, j, :] = [255, 0, 0]
                else:
                    img_train_test[i, j, :] = [255, 255, 255]

    # img[np.where(img == 255)] = 1
    # print(img.shape)
    # print(np.bincount(img.flatten()))

    imageio.imwrite(output_image_path, img_train_test)


def generate_ground_truth(img, output_image_path):
    h, w = img.shape
    img_train_test = np.zeros([3223, 7300, 3], dtype=np.uint8)
    # 8777, 12148
    # 3223, 7300

    for i in range(h-4000, h-777):
        for j in range(w-8100, w-800):
            if img[i, j] == 255 and i > 5050:
                img_train_test[i-(h-4000), j-(w-8100), :] = [255, 255, 255]

    imageio.imwrite(output_image_path, img_train_test)


def cut_image(img, output_image_path):
    imageio.imwrite(output_image_path, img[8777-4000:8777-777, 12148-8100:12148-800])


def main():
    # input_image_path = 'C:\\Users\\keill\\Desktop\\sequoia_raster_one_band.tif'
    input_image_path = 'C:\\Users\\keill\\Desktop\\gt.png'
    output_image_path = 'C:\\Users\\keill\\Desktop\\gt_rs.png'
    operation = 'resize_image'

    img = imageio.imread(input_image_path)
    print(img.shape, np.bincount(img.flatten()))

    if operation == 'train_test_image':
        save_train_test_image(img, output_image_path)
    elif operation == 'cut_image':
        cut_image(img, output_image_path)
    elif operation == 'generate_ground_truth':
        generate_ground_truth(img, output_image_path)
    elif operation == 'resize_image':
        resize_image(input_image_path, output_image_path)


if __name__ == "__main__":
    main()
