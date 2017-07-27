#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-27 上午11:18
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cv2
from tqdm import tqdm
import numpy as np
from conf.configure import Configure


def aug_test_image(src_img_dir, dest_img_dir):
    image_names = os.listdir(src_img_dir)

    if not os.path.exists(dest_img_dir):
        os.mkdir(dest_img_dir)

    for i in tqdm(range(len(image_names))):
        image_name = image_names[i]
        if 'image_augmentation' in image_name:
            continue

        img = cv2.imread(src_img_dir + image_name)

        # save primary image
        save_path = dest_img_dir + image_name
        cv2.imwrite(save_path, img)

        rows, cols = img.shape[:2]

        # 水平垂直移动
        h = np.random.randint(1, 50)
        w = np.random.randint(1, 50)
        H = np.float32([[1, 0, h], [0, 1, w]])
        res = cv2.warpAffine(img, H, (rows, cols))  # 需要图像、变换矩阵、变换后的大小
        save_path = dest_img_dir + 'aug_translate_{}_{}_'.format(h, w) + image_name
        cv2.imwrite(save_path, res)

        # 旋转
        rotate = np.random.randint(1, 359)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)
        res = cv2.warpAffine(img, M, (rows, cols))
        save_path = dest_img_dir + 'aug_rotate_{}'.format(rotate) \
                    + '_' + image_name
        cv2.imwrite(save_path, res)

        # 旋转
        rotate = np.random.randint(1, 359)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)
        res = cv2.warpAffine(img, M, (rows, cols))
        save_path = dest_img_dir + 'aug_rotate_{}'.format(rotate) \
                    + '_' + image_name
        cv2.imwrite(save_path, res)

        # 图像水平翻转
        res = cv2.flip(img, 1)
        save_path = dest_img_dir + 'aug_flip_horizontal_' + image_name
        cv2.imwrite(save_path, res)

        # 图像垂直翻转
        res = cv2.flip(img, 0)
        save_path = dest_img_dir + 'aug_flip_vertical_' + image_name
        cv2.imwrite(save_path, res)

def do_aug_test_image(image_size):
    print 'augment test images...'
    src_path = Configure.testable_test_img_path.format(image_size)
    dest_img_dir = src_path + 'image_augmentation/'
    aug_test_image(src_path, dest_img_dir)
    print 'generated {} test images'.format(len(os.listdir(dest_img_dir)))


def main():
    do_aug_test_image(224)


if __name__ == '__main__':
    main()