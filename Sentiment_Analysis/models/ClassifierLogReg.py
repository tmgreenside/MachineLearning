import keras, sys, os, csv, pickle, numpy
from keras import optimizers
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn import datasets
from copy import deepcopy

from . import Classifier
from logger import *

class ClassifierFeedFwd(Classifier.Classifier):
    def __init__(self):
        pass

    def buildModel(self):
        pass

    def getModel(self):
        return model

    def experiment(self):
        (x_train, y_train), (x_test, y_test) = self.getData()
        
