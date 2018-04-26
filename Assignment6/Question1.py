
import numpy as np
import math
from sklearn.metrics import accuracy_score

def calSigmoid(val):
    sigmoid = (1 / (1 + np.exp(val * -1)))
    return sigmoid

def calLossFunc(X, y, w_curr):
    loss = 0
    o = np.zeros(len(w_curr))
    for idx in range(len(X)):
        sig = 0 if calSigmoid(np.dot(X[idx], w_curr[idx]))<0.5 else 1
        o[idx] = sig
        loss = loss + math.pow((y[idx] - o[idx]), 2)
    return loss, o

def getWeightsGradDecent(X, y, w, learning_rate, tolerance, maxIter):
    loss = 0
    loss_new = 0
    for iter in range(maxIter):
        for n in range(len(X)):
            for wIdx in range(len(w)):
                temp, o = calLossFunc(X[n], y[n], w[wIdx])
                loss_new = loss_new + temp
                w[wIdx] = w[wIdx] + (learning_rate * np.matmul(o.T, (np.ones(len(w[wIdx])) - o)) * np.dot(X[n], w[wIdx]))
        loss = loss_new
        if abs(loss_new - loss) <= tolerance:
            return w
    return w

def getPredictedOutputAndWeightsGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits, maxIter):
    w = []
    for val in range(noOfHiddenUnits):
        w.append(np.random.randn(len(X[0]), 1))
    w = getWeightsGradDecent(X, y, w, learning_rate, tolerance, maxIter)
    y_pred = np.zeros((len(y), len(y[0])))
    for rowIndex in range(len(X)):
        temp = np.zeros(len(y[0]))
        for w_currIdx in range(len(w)):
            xNew = []
            for ls in X[rowIndex]:
                xNew.append([ls])
            xNew = np.array(xNew)
            mulArr = np.prod((xNew, w[w_currIdx]), axis=0)
            mulArrNew = []
            for eleIdx in range(len(mulArr)):
                sig = 0 if (mulArr[eleIdx] < 0 or mulArr[eleIdx] == -0) else 1
                mulArrNew.append(sig)
            temp = np.sum((temp, mulArrNew), axis=0)
        temp[temp > 0] = 1
        y_pred[rowIndex] = temp
    return y_pred, w

def evaluate_algorithmGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits, maxIter):
    y_pred, w_train = getPredictedOutputAndWeightsGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits, maxIter)
    print("accuracy score = ",accuracy_score(y, y_pred)*100)

def main():
    X = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]])
    y = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]])
    # evaluate_algorithmGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits, maxIter)
    evaluate_algorithmGradDecent(X, y, 0.00001, 0.0003, 3, 1000)

if __name__ == "__main__": main()