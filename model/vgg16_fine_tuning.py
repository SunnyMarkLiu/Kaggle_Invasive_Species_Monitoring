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
from keras.layers import Input, Dropout, Flatten, Dense
from keras import optimizers
from keras import applications
from keras import backend as K
from keras.utils import plot_model
from utils import data_util
import keras


def main():
    data_wapper = data_util.DataWapper(image_size=224)

    if K.image_data_format() == 'channels_first':  # theano
        input_shape = (3, 224, 224)
    else:  # tensorflow
        input_shape = (224, 224, 3)

    print 'built vgg16 model'
    # build the VGG16 network
    image_input = Input(shape=input_shape)
    model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=image_input)
    print('Model loaded.')
    for layer in model.layers:
        print 'frozen layer', layer
        layer.trainable = False

    # build a classifier model to put on top of the convolutional model
    top_model = Flatten(name='flatten')(model.output)
    top_model = Dense(256, activation='relu', name='fc1')(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(256, activation='relu', name='fc2')(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(1, activation='softmax', name='predictions')(top_model)

    model = Model(input=image_input, output=top_model, name='vgg16')

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adagrad(lr=0.03, epsilon=1e-06),
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='vgg16_model.png')

    print '========== start training =========='
    epochs = 100
    batch_size = 100
    train_x, train_y = data_wapper.load_all_data()
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model.fit(train_x, train_y, batch_size=batch_size,
              epochs=epochs, verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[earlystop])


if __name__ == '__main__':
    main()
