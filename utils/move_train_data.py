#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-15 下午6:59
"""

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from conf.configure import Configure


def load_pre_train_data():
    df_labels = pd.read_csv(Configure.train_labels_path)

    labels = []
    train_images = []
    for i in range(len(df_labels)):
        train_images.append(str(df_labels.ix[i][0]) + '.jpg')
        labels.append(df_labels.ix[i][1])

    return train_images, labels


def move_train_data():
    """将 train data 按照 labels 进行分割 """

    if len(os.listdir(Configure.train_img_path)) == 0:
        return

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

def main():
    print('将 train data 按照 labels 进行分割...')
    move_train_data()
    print('Done!')


if __name__ == '__main__':
    main()

