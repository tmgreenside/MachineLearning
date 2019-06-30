"""
Sentiment labels:
0 - positive
1 - negative
"""

import movie_reviews_data

class Classifier:
    def __init__(self):
        print("This is an abstract class.")
    def buildModel(self):
        pass
    def getData(self, max_words=7500):
        (x_train, y_train), (x_test, y_test) = movie_reviews_data.getReviewsData()
        return (x_train, y_train), (x_test, y_test)
