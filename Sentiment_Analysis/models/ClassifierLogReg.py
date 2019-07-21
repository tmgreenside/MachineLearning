from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from . import Classifier
from logger import *
from data.IMDB_movie_reviews import IMDB_movie_reviews

class ClassifierLogReg(Classifier.Classifier):
    def __init__(self):
        self.data = IMDB_movie_reviews()
        self.buildModel()

    def buildModel(self, c=0.5):
        self.model = LogisticRegression(C=c)

    def trainModel(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evalModel(self, x_test, y_test):
        acc = accuracy_score(y_test, self.model.predict(x_test))
        return acc

    def getModel(self):
        return self.model

    def experiment(self):
        (x_train, y_train), (x_test, y_test) = self.data.get_data()
        self.trainModel(x_train, y_train)
        print("Accuracy: ", self.evalModel(x_test, y_test))
