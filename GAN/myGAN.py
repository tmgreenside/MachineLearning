"""
Citations:

Urcuqui, C., & Navarro, A. (2016, April). Machine learning classifiers for
    android malware analysis. In Communications and Computing (COLCOM), 2016
    IEEE Colombian Conference on (pp. 1-6). IEEE.

Goodfello, I., Pouget-Abadie, J., et. al. (2014, June). Generative Adversarial
    Nets.
"""

import tensorflow
import keras
from keras import optimizers
import numpy as np
import sys, os
import pickle
from copy import deepcopy

import datasetandroidpermissions

NOISE_LENGTH = 10
TRAIN_PERC = 0.5

x, y = datasetandroidpermissions.load_data()
(x_train, y_train ), (x_test, y_test) = datasetandroidpermissions.training_and_test( x, y )
(x_mal, y_mal), (x_good, y_good) = datasetandroidpermissions.load_data_seperatly()

y_train_cat = keras.utils.to_categorical( y_train, 2 )
y_test_cat = keras.utils.to_categorical( y_test, 2 )

numFeatures = x.shape[1]

def buildModel():
    model = keras.models.Sequential([])
    model.add(keras.layers.Dense(1000, activation='relu', input_shape=(numFeatures,)))
    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

def showAccsByNumEpochs(max):
    epochs = [i for i in range(max+1)]
    accs = []
    for i in epochs:
        model = buildModel()
        model.fit(x_train, y_train, epochs=i)
        accs.append(model.evaluate(x_test, y_test))
    for i in range(len(epochs)):
        print(epochs[i], accs[i])

def buildGenerator():
    model = keras.models.Sequential([])
    model.add(keras.layers.Dense(256, activation='relu', input_shape=(numFeatures + NOISE_LENGTH,)))
    model.add(keras.layers.Dense(numFeatures, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='mean_squared_logarithmic_error')
    return model

def buildDiscriminator():
    model = keras.models.Sequential([])
    model.add(keras.layers.Dense(256, activation='relu', input_shape=(numFeatures,)))
    model.add(keras.layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    return model

def generate(generator, inputs):
    samples = deepcopy(inputs)
    noise = np.random.randint(0, 2, size=(len(inputs), NOISE_LENGTH))
    samples = np.concatenate((samples, noise), axis=1)
    samples = generator.predict(samples)

    return np.logical_or(samples, inputs)

def trainGenerator(generator, inputs, outputs):
    samples = deepcopy(inputs)
    noise = np.random.randint(0, 2, size=(len(inputs), NOISE_LENGTH))
    samples = np.concatenate((samples, noise), axis=1)
    generator.fit(samples, outputs, epochs=10)

def experiment(blackbox, generator, discriminator, epochs=10):
    # STEPS
    # 1. train discriminator and generator with a basic train set,
    # using output from the blackbox.
    for i in range(epochs):
        x_mal_train = x_mal[:int(len(x_mal) * TRAIN_PERC)]
        x_mal_test = x_mal[int(len(x_mal) * TRAIN_PERC):]
        samplesMal = generate(generator, x_mal_train)
        samplesGood = generate(generator, x_good)
        resultsMal = blackbox.predict(samplesMal, verbose=0)
        resultsGood = blackbox.predict(samplesGood, verbose=0)
        discriminator.fit(samplesMal, resultsMal, epochs=epochs)
        discriminator.fit(samplesGood, resultsGood, epochs=epochs)
        # TODO: train the generator.
        trainGenerator(generator, x_mal_train, samplesMal)
        trainGenerator(generator, x_good, samplesGood)

    # test
    mal = generate(generator, x_mal)
    loss, acc = blackbox.evaluate(mal, y_mal)
    print("Acc:", acc)

    for i in range(len(mal)):
        for j in range(len(mal[i])):
            if mal[i][j] < x_mal[i][j]:
                print("Invalid. i = ", i)
                break
    print("Valid")
    return

if __name__ == "__main__":
    print("Taking a shot at MalGAN.")
    blackbox = buildModel()
    blackbox.fit(x_train, y_train, epochs=20)
    loss, acc = blackbox.evaluate(x_test, y_test)
    print(acc)

    generator = buildGenerator()
    discriminator = buildDiscriminator()
    experiment(blackbox, generator, discriminator)
