import keras, sys, os, csv, pickle, numpy
from keras import optimizers
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn import datasets
from copy import deepcopy

from . import Classifier
from logger import *

class ClassifierFeedFwd(Classifier.Classifier):
    def __init__(self):
        self.batch_size = 64
        self.num_epochs = 2
        self.embedding_size = 32
        self.num_words = 12000
        self.dropout_rate = 0.2

        self.model = self.buildModel()

        logger.log("Feed Forward model built.")

    # model definition taken from
    # https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e
    def buildModel(self):
        model = keras.models.Sequential()
        model.add(Dense(self.num_words, input_shape=(self.num_words, ), activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def getModel(self):
        return model

    def experiment(self):
        (x_train, y_train), (x_test, y_test) = self.getData()
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self.num_epochs)
        _, acc = self.model.evaluate(x_test, y_test)
        logger.log("Accuracy: {}".format(acc))
        return

    def hyperparameter_tuning(params_set):

        return
