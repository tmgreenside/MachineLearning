import numpy as np
import sys, os
import matplotlib.pyplot as plt

def getData(filename):
    with open(filename, 'r') as csvfile:
        values = np.genfromtxt(csvfile, delimiter=',')
        values = values[1:].transpose()
        return values

# source: https://www.geeksforgeeks.org/linear-regression-python-implementation/
# alternative: handled by SciKit Learn
def getCoeff(x_val, y_val):
    n = np.size(x_val)
    # mean of x and y vector
    m_x, m_y = np.mean(x_val), np.mean(y_val)
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y_val * x_val) - n*m_y*m_x
    SS_xx = np.sum(x_val*x_val) - n*m_x*m_x

    # calculating regression coefficients
    m = SS_xy / SS_xx
    b = m_y - m*m_x

    return (m, b)

def plotData(values):
    plt.plot(values[0], values[1])
    plt.ylabel("Life Expectancy (years)")
    plt.xlabel("Year")
    plt.show()
    return

def predict(slope, intercept, x):
    return (slope * x) + intercept

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: regressionUKHealth.py <csv file>")
        exit(0)
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print("Invalid input: file does not exist.")
        exit(0)
    values = getData(filename)
    print(values)
    # x_train = values[0][:len(values[0]) - 4]
    # y_train = values[1][:len(values[0]) - 4]
    # x_test = values[0][len(values[0]) - 4:]
    # y_test = values[0][len(values[0]) - 4:]
    plotData(values)

    slope, intercept = getCoeff(values[0], values[1])
    print(slope, intercept)
