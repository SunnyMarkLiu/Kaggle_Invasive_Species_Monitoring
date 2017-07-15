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

    train_img_path = module_path + "/input/train/"
    train_labels_path = module_path + "/input/train_labels.csv"
    test_img_path = module_path + "/input/test/"

    train_labels_0_img_path = module_path + "/input/train/labels_0/"
    train_labels_1_img_path = module_path + "/input/train/labels_1/"

    submission_path = '../result/submission_{}.csv'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))
