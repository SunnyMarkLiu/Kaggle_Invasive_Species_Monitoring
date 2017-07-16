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

import cv2
from conf.configure import Configure


def image_data_preprocess(image_dir, image_size, target='train', label=None):
    """image reseize & centering & crop"""

    image_names = os.listdir(image_dir)

    for i, image_name in enumerate(image_names):
        # read image
        img = cv2.imread(image_dir + image_name, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if target == 'train':
            if label == 0:
                save_dir = Configure.trainable_train_labels_0_img_path.format(image_size)
            else:
                save_dir = Configure.trainable_train_labels_1_img_path.format(image_size)
        else:
            save_dir = Configure.testable_test_img_path.format(image_size)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # resize
        img = cv2.resize(img, (image_size, image_size))
        # save image
        cv2.imwrite(save_dir + image_name, img)


def main():
    print('image reseize & centering & crop...')
    image_size = Configure.vgg_image_size
    image_data_preprocess(Configure.train_labels_1_img_path, 'train', 0)
    image_data_preprocess(Configure.train_labels_0_img_path, 'train', 1)
    image_data_preprocess(Configure.test_img_path, image_size, 'test')
    print('Done!')


if __name__ == '__main__':
    main()
