#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-20 上午10:46
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import cv2
from conf.configure import Configure

def load_pre_train_data():
    df_labels = pd.read_csv(Configure.train_labels_path)

    labels = []
    train_images = []
    for i in range(len(df_labels)):
        train_images.append(str(df_labels.ix[i][0]) + '.jpg')
        labels.append(df_labels.ix[i][1])

    return train_images, labels


def load_pre_test_data():
    sample_submission = pd.read_csv("../input/sample_submission.csv")

    test_names = []
    test_images = []

    for i in range(len(sample_submission)):
        test_images.append(Configure.test_img_path + str(int(sample_submission.ix[i][0])) + '.jpg')
        test_names.append(sample_submission.ix[i][0])

    return test_images, test_names


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


def move_train_data():
    """将 train data 按照 labels 进行分割 """

    import shutil
    train_images, labels = load_pre_train_data()
    df_train_images = pd.DataFrame({"train_images": train_images,
                                    "labels": labels})
    labels_1_images = df_train_images["train_images"][df_train_images["labels"] == 1].values
    labels_0_images = df_train_images["train_images"][df_train_images["labels"] == 0].values

    # 移动到不同的目录
    print("移动 labels_1")
    for img in labels_1_images:
        shutil.move(Configure.train_img_path + img, Configure.train_labels_1_img_path + img)
    print("移动 labels_0")
    for img in labels_0_images:
        shutil.move(Configure.train_img_path + img, Configure.train_labels_0_img_path + img)


def image_augmentor():
    """训练集数据扩充"""


def main():
    print('将 train data 按照 labels 进行分割...')
    move_train_data()
    # print('训练集数据扩充...')
    # image_augmentor()
    print('image reseize & centering & crop...')
    image_data_preprocess(Configure.train_labels_1_img_path)
    image_data_preprocess(Configure.train_labels_0_img_path)
    image_data_preprocess(Configure.test_img_path)
    print('Done!')


if __name__ == '__main__':
    main()
