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
    def get_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
