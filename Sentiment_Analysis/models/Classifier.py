"""
Sentiment labels:
0 - positive
1 - negative
"""

class Classifier:
    def __init__(self):
        self.model = None
        raise NotImplementedError()

    def buildModel(self):
        raise NotImplementedError()

    def getModel(self):
        return self.model
