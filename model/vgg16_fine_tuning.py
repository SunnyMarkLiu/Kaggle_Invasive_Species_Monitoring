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
from keras import optimizers, regularizers
from keras import applications
from keras import backend as K
from keras.utils import plot_model
from sklearn.metrics import roc_auc_score
# from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import keras
from utils import data_util
import pandas as pd
from sklearn.model_selection import train_test_split
from conf.configure import Configure


def main():
    image_size = 224
    # all train data
    train_x_image_path, train_y = data_util.load_train_data(image_size=image_size)
    # split train/validate
    train_X, validate_X, train_y, validate_y = train_test_split(train_x_image_path,
                                                                train_y,
                                                                test_size=0.1,
                                                                random_state=0)
    train_data_wapper = data_util.DataWapper(train_X, train_y, istrain=True)
    validate_data_wapper = data_util.DataWapper(validate_X, validate_y, istrain=True)

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
    top_model = Dense(256, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(5e-4))(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(256, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(5e-4))(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(1, activation='sigmoid', name='predictions')(top_model)

    model = Model(input=image_input, output=top_model, name='vgg16')

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='vgg16_model.png')

    print '========== start training =========='
    print 'training data size: ', train_X.shape[0]
    print 'validate data size: ', validate_X.shape[0]

    epochs = 100
    batch_size = 50
    validate_X, validate_y = validate_data_wapper.load_all_data()

    def data_generator(gen_batch_size):
        while 1:
            batch_x, batch_y = train_data_wapper.next_batch(gen_batch_size)
            yield batch_x, batch_y

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(Configure.vgg16_best_model_weights,
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    model.fit_generator(
        data_generator(gen_batch_size=batch_size),
        steps_per_epoch=train_X.shape[0] // batch_size,
        epochs=epochs, verbose=1,
        validation_data=(validate_X, validate_y),
        callbacks=[earlystop, checkpoint]
    )

    print '============ load weights ============'
    model.load_weights(Configure.vgg16_best_model_weights)
    print '========== start validating =========='
    predict = model.predict(validate_X, batch_size=100, verbose=1)
    val_roc = roc_auc_score(validate_y, predict)
    print 'validate roc_auc_score =', val_roc

    print '========== start predicting =========='
    # predict
    # all test data
    test_image_name, test_x = data_util.load_test_data(image_size)
    test_data_wapper = data_util.DataWapper(test_x, istrain=False)
    test_x, _ = test_data_wapper.load_all_data()

    predict = model.predict(test_x, batch_size=100, verbose=1)
    predict = predict[:, 0]
    predict_df = pd.DataFrame({'name': test_image_name,
                               'invasive': predict})
    predict_df = predict_df[['name', 'invasive']]
    predict_df.to_csv(Configure.submission_path, index=False)

if __name__ == '__main__':
    main()
