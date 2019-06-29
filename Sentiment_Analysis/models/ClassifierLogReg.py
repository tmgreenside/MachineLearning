from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
