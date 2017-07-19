#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-17 下午5:46
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cv2
from tqdm import tqdm
import numpy as np
from conf.configure import Configure


def aug_train_image(src_img_dir, dest_img_dir):

    image_names = os.listdir(src_img_dir)

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
        H = np.float32([[1, 0, np.random.randint(1, 50)], [0, 1, np.random.randint(1, 50)]])
        res = cv2.warpAffine(img, H, (rows, cols))  # 需要图像、变换矩阵、变换后的大小
        save_path = dest_img_dir + 'aug_translate_' + image_name
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

def main():

    print 'augment train label 0 data...'
    src_path = Configure.trainable_train_labels_0_img_path.format(224)
    dest_img_dir = src_path + 'image_augmentation/'
    aug_train_image(src_path, dest_img_dir)
    print 'generate {} train label 0 data'.format(len(os.listdir(dest_img_dir)))

    print 'augment train label 1 data...'
    src_path = Configure.trainable_train_labels_1_img_path.format(224)
    dest_img_dir = src_path + 'image_augmentation/'
    aug_train_image(src_path, dest_img_dir)
    print 'generate {} train label 1 data'.format(len(os.listdir(dest_img_dir)))

if __name__ == '__main__':
    main()
