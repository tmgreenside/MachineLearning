from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from . import Classifier
from logger import *

class ClassifierLogReg(Classifier.Classifier):
    def __init__(self):
        self.model = self.buildModel()
        logger.log("Logistic regression model build.")

    def buildModel(self):
        model = LogisticRegression(C=0.05)
        return model

    def getModel(self):
        return self.model

    def experiment(self):
        (x_train, y_train), (x_test, y_test) = self.getData()
        self.model.fit(x_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(x_test))
        logger.log("Accuracy: {}".format(acc))
        return
