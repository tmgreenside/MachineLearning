import keras, sys, os, csv, pickle, numpy
from keras import optimizers
from keras.layers import Flatten, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from sklearn import datasets
from copy import deepcopy

from . import Classifier
from logger import *
from data.IMDB_movie_reviews import IMDB_movie_reviews

class ClassifierConvNN(Classifier.Classifier):
    def __init__(self, model_file=None, percent_train=0.5):
        self.batch_size = 64
        self.num_epochs = 1
        self.embedding_size = 32
        self.num_words = 12000
        self.dropout_rate = 0.5
        self.num_filters = 64
        self.kernel_size = 5

        if model_file==None:
            self.model = self.buildModel()
        else:
            raise NotImplementedError()
        self.data = IMDB_movie_reviews(percent_train)

    def buildModel(self):
        model = keras.models.Sequential()
        model.add(Conv1D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            activation='relu',
            input_shape=(self.num_words,1)
        ))
        model.add(BatchNormalization(axis=1))
        model.add(Conv1D(filters=self.num_filters,
            kernel_size=self.kernel_size,
            activation='relu'
        ))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(self.dropout_rate))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def experiment(self):
        (x_train, y_train), (x_test, y_test) = self.data.get_data()
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)
        x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self.num_epochs)
        _, acc = self.model.evaluate(x_test, y_test)
        logger.log("Accuracy: {}".format(acc))
        return
