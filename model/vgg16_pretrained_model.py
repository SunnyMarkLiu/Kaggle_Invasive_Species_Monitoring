#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-16 下午2:11
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras import applications
from keras import backend as K
from keras.utils import plot_model
from utils import data_util


def main():
    data_wapper = data_util.DataWapper(image_size=224)

    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)

    print 'built vgg16 model'
    # build the VGG16 network
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Flatten(name='flatten', input_shape=model.output_shape[1:])(model.output)
    top_model = Dense(256, activation='relu', name='fc1')(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(256, activation='relu', name='fc2')(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(1, activation='softmax', name='predictions')(top_model)

    model = Model(model.input, top_model, name='vgg16')
    # set the first 19 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:19]:
        print 'frozen layer', layer
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    plot_model(model, to_file='vgg16_model.png')

    print '========== start training =========='


if __name__ == '__main__':
    main()
