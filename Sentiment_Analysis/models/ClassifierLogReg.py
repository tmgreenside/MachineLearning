from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
<<<<<<< HEAD
=======
from sklearn.model_selection import train_test_split
>>>>>>> b98ca39a6b91f546ff238bb8c46bdbf11f3e8fdb

from . import Classifier
from logger import *
from data.IMDB_movie_reviews import IMDB_movie_reviews

class ClassifierLogReg(Classifier.Classifier):
    def __init__(self):
<<<<<<< HEAD
        self.data = IMDB_movie_reviews()
        self.buildModel()

    def buildModel(self, c=0.5):
        self.model = LogisticRegression(C=c)

    def trainModel(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evalModel(self, x_test, y_test):
        acc = accuracy_score(y_test, self.model.predict(x_test))
        return acc
=======
        self.model = self.buildModel()
        logger.log("Logistic regression model built.")

    def buildModel(self):
        model = LogisticRegression(C=0.05)
        return model
>>>>>>> b98ca39a6b91f546ff238bb8c46bdbf11f3e8fdb

    def getModel(self):
        return self.model

    def experiment(self):
<<<<<<< HEAD
        (x_train, y_train), (x_test, y_test) = self.data.get_data()
        self.trainModel(x_train, y_train)
        print("Accuracy: ", self.evalModel(x_test, y_test))
=======
        (x_train, y_train), (x_test, y_test) = self.getData()
        self.model.fit(x_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(x_test))
        logger.log("Accuracy: {}".format(acc))
        return
>>>>>>> b98ca39a6b91f546ff238bb8c46bdbf11f3e8fdb
