# Data source: http://ai.stanford.edu/~amaas/data/sentiment/

from enum import Enum
import csv, pickle, numpy, os, keras

from .data import Data

class Sentiment(Enum):
    NEG = 0
    POS = 1

VOCAB_FILE = "aclIMDB/imdb.vocab"
DATAPATH_TRAIN = os.getcwd() + "/aclImdb/train/"
DATAPATH_TEST = os.getcwd() + "/aclImdb/test/"
MAX_WORDS = 12000

POS = "pos.dat"
NEG = "neg.dat"

class IMDB_movie_reviews(Data):
    def __init__(self, max_words=MAX_WORDS, percent_train=0.25):
        self.max_words = max_words
        self.build_dict()
        if len(self.dict) < self.max_words:
            self.max_words = len(self.dict)
        self.load_data()

    # Retrieves data from the vectorized files, or parses the original text if
    # the vectors are not available.
    def load_data(self):
        if not os.path.isfile(POS):
            self.vectorizeWordIndex(0, max_words=MAX_WORDS)
        if not os.path.isfile(NEG):
            self.vectorizeWordIndex(1, max_words=MAX_WORDS)
        print("Loading data. This may take a while.")
        self.x_pos, self.y_pos = pickle.load(open(POS, "rb"))
        self.x_neg, self.y_neg = pickle.load(open(NEG, "rb"))
        print("Data loaded.")

        self.x = numpy.concatenate((self.x_pos, self.x_neg))
        self.y = numpy.concatenate((self.y_pos, self.y_neg))

        self.x_train = numpy.concatenate((self.x[:int(len(self.x)/4)], self.x[int(3 * len(self.x) / 4):]))
        self.y_train = numpy.concatenate((self.y[:int(len(self.y)/4)], self.y[int(3 * len(self.y) / 4):]))
        self.x_test = self.x[int(len(self.x)/4):int(3 * len(self.x) / 4)]
        self.y_test = self.y[int(len(self.y)/4):int(3*len(self.y)/4)]

    def get_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    # This function returns the list of words in array form.
    def build_dict(self):
        self.dict = []
        with open(VOCAB_FILE, "r") as vocabFile:
            word = vocabFile.readline()
            while word != "":
                self.dict.append(word.rstrip())
                word = vocabFile.readline()

    def get_dict(self):
        return self.dict

    # This function saves an x and y array representing either the positive
    # or negative review set, with the x array being 2D where each index
    # represents a word in the dictionary, with the value being the word's
    # number of occurences.
    # Note: does not work for very large datasets.
    def vectorizeWordIndex(self, output, reduced=True, max=5000, max_words=10000):
        # TODO. Feel free to change parameters
        if output == 0:
            path = DATAPATH_TRAIN + "pos/"
        else:
            path = DATAPATH_TRAIN + "neg/"
        files = os.listdir(path)
        if reduced == False or len(DICT) < max_words:
            vectorLength = len(files)
        else:
            vectorLength = max
        vectors = numpy.zeros((vectorLength, max_words))
        if output == 0:
            y_vector = numpy.zeros(vectorLength)
        else:
            y_vector = numpy.ones(vectorLength)
        for i in range(vectorLength):
            print("Vectorizing file {} of {}".format(i+1, vectorLength))
            wordCounts = {}
            with open(path + files[i], "r") as next_file:
                lines = next_file.readlines()
                for line in lines:
                    for word in line.split():
                        if word in wordCounts:
                            wordCounts[word] += 1
                        else:
                            wordCounts[word] = 1

            for j in range(max_words):
                if DICT[j] in wordCounts:
                    vectors[i][j] += wordCounts[DICT[j]]

        if output == 0:
            pickle.dump((vectors, y_vector), open("pos.dat", "wb"))
        else:
            pickle.dump((vectors, y_vector), open("neg.dat", "wb"))

if __name__ == "__main__":
    train, test = vectorizeWordIndex(getDict(), )
