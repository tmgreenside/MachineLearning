# Data source: http://ai.stanford.edu/~amaas/data/sentiment/

from enum import Enum
import csv, pickle, numpy, os, keras

class Sentiment(Enum):
    NEG = 0
    POS = 1

VOCAB_FILE = "aclIMDB/imdb.vocab"
DATAPATH_TRAIN = os.getcwd() + "/aclImdb/train/"
DATAPATH_TEST = os.getcwd() + "/aclImdb/test/"
MAX_WORDS = 12000

POS = "pos.dat"
NEG = "neg.dat"

# This function returns the list of words in array form.
def getDict():
    dict = []
    with open(VOCAB_FILE, "r") as vocabFile:
        word = vocabFile.readline()
        while word != "":
            dict.append(word.rstrip())
            word = vocabFile.readline()
    return dict

DICT = getDict()
if len(DICT) < MAX_WORDS:
    MAX_WORDS = len(DICT)


# This function saves an x and y array representing either the positive
# or negative review set, with the x array being 2D where each index
# represents a word in the dictionary, with the value being the word's
# number of occurences.
# Note: does not work for very large datasets.
def vectorizeWordIndex(output, reduced=True, max=5000, max_words=10000):
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

# Retrieves data from the vectorized files, or parses the original text if
# the vectors are not available.
def getReviewsData():
    if not os.path.isfile(POS):
        vectorizeWordIndex(0, max_words=MAX_WORDS)
    if not os.path.isfile(NEG):
        vectorizeWordIndex(1, max_words=MAX_WORDS)
    print("Loading data. This may take a while.")
    x_pos, y_pos = pickle.load(open(POS, "rb"))
    x_neg, y_neg = pickle.load(open(NEG, "rb"))
    print("Data loaded.")

    x = numpy.concatenate((x_pos, x_neg))
    y = numpy.concatenate((y_pos, y_neg))

    x_train = numpy.concatenate((x[:int(len(x)/4)], x[int(3 * len(x) / 4):]))
    y_train = numpy.concatenate((y[:int(len(y)/4)], y[int(3 * len(y) / 4):]))
    x_test = x[int(len(x)/4):int(3 * len(x) / 4)]
    y_test = y[int(len(y)/4):int(3*len(y)/4)]

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    train, test = vectorizeWordIndex(getDict(), )
