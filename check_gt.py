import imageio
import numpy as np
import argparse


def get_heatmap(x, y, h, w, sigma):
    grid_x = np.arange(h)
    grid_y = np.arange(w)
    double_sigma2 = 2 * sigma * sigma
    heatmap = np.zeros((h, w), dtype=np.float)
    for i in range(len(x)):
        exp_x = np.exp(-(grid_x-x[i])**2/double_sigma2)
        exp_y = np.exp(-(grid_y-y[i])**2/double_sigma2)
        exp = np.outer(exp_x, exp_y)
        heatmap = np.maximum(heatmap, exp)
    return heatmap


def main():
    parser = argparse.ArgumentParser(description='check_gt')
    # general options
    parser.add_argument('--input', type=str, required=True, help='input path')
    parser.add_argument('--output', type=str, required=True, help='output path')
    args = parser.parse_args()
    print(args)

    img = imageio.imread(args.input)
    print(img.shape)
    print(np.bincount(img.astype(int).flatten()))

    h, w = img.shape
    x, y = np.where(img == 1)

    print(x.shape, y.shape)

    heat_map = get_heatmap(x, y, h, w, sigma=1.0)
    imageio.imwrite(args.output, heat_map)


if __name__ == "__main__":
    main()
