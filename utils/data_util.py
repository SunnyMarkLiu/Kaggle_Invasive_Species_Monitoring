#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-20 上午10:46
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import cv2
from conf.configure import Configure


def centering_image(img):
    size = [256, 256]
    img_size = img.shape[:2]
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img
    return resized


def image_data_preprocess(image_dir):
    """image reseize & centering & crop"""

    image_paths = os.listdir(image_dir)

    for i, image_path in enumerate(image_paths):
        image_file = image_dir + image_path
        # read image
        img = cv2.imread(image_dir + image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize
        if img.shape[0] > img.shape[1]:
            tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
        else:
            tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

        # centering
        img = centering_image(cv2.resize(img, dsize=tile_size))

        # output 224*224px
        img = img[16:240, 16:240]

        # save image
        cv2.imwrite(image_file, img)


def main():
    print('image reseize & centering & crop...')
    image_data_preprocess(Configure.train_labels_1_img_path)
    image_data_preprocess(Configure.train_labels_0_img_path)
    image_data_preprocess(Configure.test_img_path)
    print('Done!')


if __name__ == '__main__':
    main()
