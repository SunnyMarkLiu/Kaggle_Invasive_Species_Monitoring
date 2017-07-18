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


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def transform_image(img, ang_range, shear_range, trans_range, brightness=0):
    """
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    """
    # Rotation

    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    # Brightness


    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    if brightness == 1:
        img = augment_brightness_camera_images(img)

    return img


def aug_train_image(src_img_dir, dest_img_dir, augment_multiple):

    image_names = os.listdir(src_img_dir)

    for i in tqdm(range(len(image_names))):
        image_name = image_names[i]
        if 'image_augmentation' in image_name:
            continue

        img = cv2.imread(src_img_dir + image_name)

        # save primary image
        save_path = dest_img_dir + image_name
        cv2.imwrite(save_path, img)

        for j in range(augment_multiple + 1):
            img = transform_image(img, 20, 10, 5, brightness=1)
            save_path = dest_img_dir + 'aug_' + str(j+1) + '_' + image_name
            cv2.imwrite(save_path, img)


def main():

    print 'augment train label 0 data...'
    src_path = Configure.trainable_train_labels_0_img_path.format(224)
    dest_img_dir = src_path + 'image_augmentation/'
    aug_train_image(src_path, dest_img_dir, augment_multiple=10)
    print 'generate {} train label 0 data'.format(len(os.listdir(dest_img_dir)))

    print 'augment train label 1 data...'
    src_path = Configure.trainable_train_labels_1_img_path.format(224)
    dest_img_dir = src_path + 'image_augmentation/'
    aug_train_image(src_path, dest_img_dir, augment_multiple=10)
    print 'generate {} train label 1 data'.format(len(os.listdir(dest_img_dir)))

if __name__ == '__main__':
    main()
