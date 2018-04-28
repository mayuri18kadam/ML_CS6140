
import numpy as np
import math
import random
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def calSigmoid(val):
    sigmoid = (1 / (1 + np.exp(val * -1)))
    return sigmoid

def feedForward(x_final, wAll):
    o1 = np.zeros(len(wAll[0]))
    o2 = np.zeros(len(wAll[1]))

    for i in range(len(o1)):
        prod1 = np.zeros(len(x_final))
        for j in range(len(x_final)):
            prod1[j] = x_final[j] * wAll[0][i][j]
        o1[i] = calSigmoid(np.sum(prod1))

    o1_final = []
    for ls in o1:
        o1_final.append([ls])
    o1_final = np.array(o1_final)
    for i in range(len(o2)):
        prod1 = np.zeros(len(o1_final))
        for j in range(len(o1_final)):
            prod1[j] = o1_final[j] * wAll[1][i][j]
        o2[i] = calSigmoid(np.sum(prod1))

    o2_final = []
    for ls in o2:
        o2_final.append([ls])
    o2_final = np.array(o2_final)

    return o1_final, o2_final

def backPropogate(x_final, y_final, wAll, learning_rate, o1_final, o2_final):
    # updating weights between hidden layer and output layer
    for idx1 in range(len(wAll[1])):
        for idx2 in range(len(wAll[1][idx1])):
            wAll[1][idx1][idx2] = wAll[1][idx1][idx2] + \
                                  (learning_rate * (y_final[idx1] - o2_final[idx1]) * o2_final[idx1] *
                                   (1 - o2_final[idx1]) * o1_final[idx2])

    # updating weights between hidden layer and input layer
    for idx1 in range(len(wAll[0])):
        for idx2 in range(len(wAll[0][idx1])):
            # computing downstream sum
            sum = 0
            for idx3 in range(len(wAll[1])):
                sum = sum + wAll[1][idx3][idx1]*x_final[idx2]
            # update w
            wAll[0][idx1][idx2] = wAll[0][idx1][idx2] + (learning_rate * o1_final[idx1] * (1 - o1_final[idx1]) * sum)

    return wAll

def update(x_final, y_final, learning_rate, tolerance, wAll):
    o1_final, o2_final = feedForward(x_final, wAll)
    wAll = backPropogate(x_final, y_final, wAll, learning_rate, o1_final, o2_final)
    return wAll

def getPredictedOutput(X, wAll):
    y_pred = np.zeros((len(X), len(X[0])))
    idx1 = 0
    for xRow in X:
        x_final = []
        for ls in xRow:
            x_final.append([ls])
        x_final = np.array(x_final)
        y1, y2 = feedForward(x_final, wAll)
        for i in range(len(x_final)):
            y_pred[idx1][i] = 0 if y2[i]*x_final[i]==0 else 1
        idx1 = idx1 + 1
    return y_pred

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

    for iter in range(maxIter):
        # random.shuffle(X)
        idx = 0
        for xRow in X:
            x_final = []
            for ls in xRow:
                x_final.append([ls])
            x_final = np.array(x_final)

            y_final = []
            for ls in y[idx]:
                y_final.append([ls])
            y_final = np.array(y_final)

            wAll = update(x_final, y_final, learning_rate, tolerance, wAll)
            idx = idx + 1

    y_pred = getPredictedOutput(X, wAll)
    print("y_pred = ",y_pred)




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
    score = evaluate_algorithmGradDecent(X, y, 0.001, 0.00001, 3, 1000)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('acc_score VS noOfHiddenUnits')
    # ax1.set_xlabel('noOfHiddenUnits')
    # ax1.set_ylabel('acc_score')
    # ax1.plot(noOfHiddenUnits, acc_score, ls='--', marker='o', c='m', label='acc_score')
    # plt.show()

if __name__ == "__main__": main()