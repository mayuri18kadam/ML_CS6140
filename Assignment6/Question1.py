
import numpy as np
import math
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

def getPredictedOutputAndWeightsGradDecent(X, y, learning_rate, tolerance, maxIter, w):
    w = getWeightsGradDecent(X, y, w, learning_rate, tolerance, maxIter)
    print("w = ",w)
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
                sig = 0 if (calSigmoid(mulArr[eleIdx]) == 0.5) else 1
                mulArrNew.append(sig)
            temp = np.sum((temp, mulArrNew), axis=0)
        temp[temp > 0] = 1
        y_pred[rowIndex] = temp
    print("y_pred = ",y_pred)
    return y_pred, w

def evaluate_algorithmGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits, maxIter):
    # initializations
    wAll = []
    w1 = []
    w2 = []
    for val in range(noOfHiddenUnits):
        w1.append(np.random.randn(len(X[0]), 1))
    wAll.append(w1)
    for val in range(len(X[0])):
        w2.append(np.random.randn(noOfHiddenUnits, 1))
    wAll.append(w2)



    w = getWeightsGradDecent(X, y, w, learning_rate, tolerance, maxIter)
    acc_score = []
    while True :
        y_pred, w = getPredictedOutputAndWeightsGradDecent(X, y, learning_rate, tolerance, maxIter, w)
        score = accuracy_score(y, y_pred)*100
        acc_score.append(score)
        print("accuracy score = ",score)
        if score > 95:
            break
    return score
    # return acc_score

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
    # acc_score = []
    # noOfHiddenUnits = [1,2,3,4,5,6,7,8,9,10]
    # for n in noOfHiddenUnits:
    #     # evaluate_algorithmGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits, maxIter)
    #     score = evaluate_algorithmGradDecent(X, y, 0.00001, 0.00001, 1, 1000)
    #     acc_score.append(score)
    score = evaluate_algorithmGradDecent(X, y, 0.00001, 0.00001, 3, 1000)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('acc_score VS noOfHiddenUnits')
    # ax1.set_xlabel('noOfHiddenUnits')
    # ax1.set_ylabel('acc_score')
    # ax1.plot(noOfHiddenUnits, acc_score, ls='--', marker='o', c='m', label='acc_score')
    # plt.show()

if __name__ == "__main__": main()