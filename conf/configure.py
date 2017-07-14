#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-20 下午2:17
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import time


class Configure(object):
    """global config"""

    # data
    data_base_path = '/data/sunnymarkliu/kaggle/invasive_species_monitoring'
    train_img_path = data_base_path + "/train/original/"
    train_labels_path = data_base_path + "/train/train_labels.csv"
    test_img_path = data_base_path + "/test/original/"

    train_labels_0_img_path = data_base_path + "/train/labels_0/"
    train_labels_1_img_path = data_base_path + "/train/labels_1/"

    # models
    alexnet_image_width, alexnet_image_height = 224, 224
    vgg_image_width, vgg_image_height = 224, 224

    # result
    submission_path = '../result/submission_{}.csv'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))
