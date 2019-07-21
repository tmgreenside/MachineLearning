"""
Sentiment labels:
0 - positive
1 - negative
"""

class Classifier:
    def __init__(self):
        print("This is an abstract class.")
    def buildModel(self):
        pass
<<<<<<< HEAD
    def get_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
=======
    def getData(self, max_words=7500):
        (x_train, y_train), (x_test, y_test) = movie_reviews_data.getReviewsData()
        return (x_train, y_train), (x_test, y_test)
>>>>>>> b98ca39a6b91f546ff238bb8c46bdbf11f3e8fdb
