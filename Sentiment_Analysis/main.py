"""
Sentiment labels:
0 - positive
1 - negative
"""
import sys, os
from models.ClassifierFeedFwd import ClassifierFeedFwd

from logger import *
from models import ClassifierFeedFwd

def experiment():
    logger.log("Working with Feed Forward model")
    modelFwd = ClassifierFeedFwd.ClassifierFeedFwd()
    modelFwd.experiment()

if __name__ == "__main__":
    try:
        experiment()
    except KeyboardInterrupt:
        logger.log("Interrupted: KeyboardInterrupt. Aborting.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
