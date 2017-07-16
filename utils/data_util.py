#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-20 下午2:11
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np

from conf.configure import Configure


def load_train_data(image_size):
    """加载处理后的训练集"""
    train_x = []

    base_path = Configure.trainable_train_labels_0_img_path.format(image_size)
    images = os.listdir(base_path)
    images = [base_path + image for image in images]
    train_x.extend(images)
    train_y = [0] * len(images)

    base_path = Configure.trainable_train_labels_1_img_path.format(image_size)
    images = os.listdir(base_path)
    images = [base_path + image for image in images]
    train_x.extend(images)
    train_y.extend([1] * len(images))

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y = train_y.reshape((train_y.shape[0], 1))

    return train_x, train_y


class DataWapper(object):
    def __init__(self, image_size):
        self.image_size = image_size
        train_x, train_y = load_train_data(image_size=224)
        self.x = train_x
        self.y = train_y
        self.pointer = 0
        self.total_count = self.x.shape[0]
        self.shuffle()

    def shuffle(self):
        shuffled_index = np.arange(0, self.total_count)
        np.random.shuffle(shuffled_index)
        self.x = self.x[shuffled_index]
        self.y = self.y[shuffled_index]

    def load_all_data(self):
        return self.next_batch(self.x.shape[0])

    def next_batch(self, batch_size):
        end = self.pointer + batch_size
        if end > self.total_count:
            end = self.total_count

        batch_x = self.x[self.pointer: end]
        y = self.y[self.pointer: end]

        x = []
        for img_path in batch_x:
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img = img[0]
            x.append(img)

        self.pointer = end

        if self.pointer == self.total_count:
            self.shuffle()
            self.pointer = 0

        x = np.array(x)
        return x, y


def test():
    data_wapper = DataWapper(image_size=224)
    batch_x, batch_y = data_wapper.next_batch(20)

    print batch_x.shape
    print batch_y.shape
    print batch_y


if __name__ == '__main__':
    test()
