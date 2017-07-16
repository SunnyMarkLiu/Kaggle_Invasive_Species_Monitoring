#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-20 下午2:11
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import cv2

from conf.configure import Configure

imagenet_mean = {'R': 103.939,
                 'G': 116.779,
                 'B': 123.68}


def load_train_data():
    """加载处理后的训练集"""
    train_x = []
    train_y = []

    base_path = Configure.train_labels_0_img_path
    images = os.listdir(base_path)
    for image_path in images:
        image_path = base_path + image_path
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float16)
        img[:, :, 0] -= imagenet_mean['R']
        img[:, :, 1] -= imagenet_mean['G']
        img[:, :, 2] -= imagenet_mean['B']

        train_x.append(img)
        train_y.append(0)

    base_path = Configure.train_labels_1_img_path
    images = os.listdir(base_path)
    for image_path in images:
        image_path = base_path + image_path
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float16)
        img[:, :, 0] -= imagenet_mean['R']
        img[:, :, 1] -= imagenet_mean['G']
        img[:, :, 2] -= imagenet_mean['B']

        train_x.append(img)
        train_y.append(1)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y = train_y.reshape((train_y.shape[0], 1))

    return train_x, train_y


def load_test_data():
    """加载处理后的测试集"""
    test_x = []
    test_name = []

    base_path = Configure.test_img_path
    images = os.listdir(base_path)
    for image_path in images:
        test_name.append(int(image_path.split('.')[0]))
        image_path = base_path + image_path
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float16)
        img[:, :, 0] -= imagenet_mean['R']
        img[:, :, 1] -= imagenet_mean['G']
        img[:, :, 2] -= imagenet_mean['B']

        test_x.append(img)

    test_name = np.array(test_name)
    test_x = np.array(test_x)
    return test_name, test_x


class DataWapper(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pointer = 0
        self.total_count = self.x.shape[0]

    def shuffle(self):
        shuffled_index = np.arange(0, self.total_count)
        np.random.shuffle(shuffled_index)
        self.x = self.x[shuffled_index]
        self.y = self.y[shuffled_index]

    def next_batch(self, batch_size):
        end = self.pointer + batch_size
        if end > self.total_count:
            end = self.total_count

        batch_x = self.x[self.pointer: end]
        batch_y = self.y[self.pointer: end]

        self.pointer = end

        if self.pointer == self.total_count:
            self.shuffle()
            self.pointer = 0

        return batch_x, batch_y

def test():
    train_x, train_y = load_train_data()
    data_wapper = DataWapper(train_x, train_y)
    data_wapper.shuffle()

    batch_x, batch_y = data_wapper.next_batch(20)

    print(batch_x.shape)
    print(batch_y.shape)
    print(batch_y)

if __name__ == '__main__':
    test()
