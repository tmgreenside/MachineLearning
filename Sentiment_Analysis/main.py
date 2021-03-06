import sys, os
from models.ClassifierFeedFwd import ClassifierFeedFwd
from models.ClassifierLogReg import ClassifierLogReg
from models.ClassifierConvNN import ClassifierConvNN
from models.autoencoderCNN import AutoencoderCNN

from logger import *

def experiment():
    # logger.log("Working with Logistic Regression model")
    # modelFwd = ClassifierLogReg()
    # modelFwd.experiment()
    # logger.log("Working with Feed Forward model.")
    # modelFwd = ClassifierFeedFwd.ClassifierFeedFwd()
    # modelFwd.experiment()
    # logger.log("Working with Convolutional model.")
    # modelCNN = ClassifierConvNN()
    # modelCNN.experiment()
    logger.log("Experimenting with AutoEncoder")
    modelAutoEnc = AutoencoderCNN()
    modelAutoEnc.experiment()


if __name__ == "__main__":
    try:
        experiment()
    except KeyboardInterrupt:
        logger.log("Interrupted: KeyboardInterrupt. Aborting.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
