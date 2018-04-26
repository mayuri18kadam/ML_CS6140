
import numpy as np
import random
import math
from math import log1p
# import matplotlib.pyplot as plt

def calSigmoid(val):
    sigmoid = (1 / (1 + np.exp(val * -1)))
    return sigmoid

def calLossFunc(X, y, w_curr):
    loss = 0
    o = []
    idx = 0
    for x,w  in X,w_curr:
        sig = calSigmoid(np.matmul(x, w))
        o.append(sig)
        loss = loss + math.pow((y[idx] - (0 if sig<0.5 else 1)), 2)
        idx = idx + 1
    print("o = ",o)
    return loss, o


def getWeightsGradDecent(X, y, w, learning_rate, tolerance):
    loss = 0
    loss_new = 0
    for n in range(len(X)):
        for wIdx in range(len(w)):
            temp, o = calLossFunc(X[n], y[n], w[wIdx])
            loss_new = loss_new + temp
            w[wIdx] = w[wIdx] + learning_rate * np.dot(o, o) * np.matmul(np.transpose(X), updated))
        loss = loss_new
        if abs(loss_new - loss) <= tolerance:
            return w
    return w


def getPredictedOutputAndWeightsGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits):
    w = []
    for val in range(noOfHiddenUnits):
        w.append(np.random.randn(len(X[0]), 1))
    w = getWeightsGradDecent(X, y, w, learning_rate, tolerance)

    y_pred = []
    # for rowIndex in range(len(X)):
    #     y_pred.append(calSigmoid(np.matmul(X_final[rowIndex], w)))
    return y_pred, w

def evaluate_algorithmGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits):
    y_pred_train, w_train = getPredictedOutputAndWeightsGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits)
    # y_pred_train_final = []
    # for y_data in y_pred_train:
    #     y_pred_train_final.append(0) if y_data < 0.5 else y_pred_train_final.append(1)
    #
    # X_test_final = np.append(np.ones((len(xTest), 1)), xTest, axis=1)
    # y_pred_test = np.matmul(X_test_final, w_train)
    # y_pred_test_final = []
    # for y_data in y_pred_test:
    #     y_pred_test_final.append(0) if y_data < 0.5 else y_pred_test_final.append(1)


def main():

    X = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]]
    y = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]]
    # evaluate_algorithmGradDecent(X, y, learning_rate, tolerance, noOfHiddenUnits)
    evaluate_algorithmGradDecent(X, y, 0.00001, 0.0003, 3)

    #     for t in tolerance:
    #         print("*********************************************************************************")
    #         print("tolerance = ",t)
    #         print("*********************************************************************************")
    #         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
    #         rmse_train, loss = evaluate_algorithmGradDecent(breastcancer.values, 10, 1000, 0.0004, t)
    #         lossArr.append(loss)
    #         rmseArr.append(rmse_train)
    #         lossArrMin.append(np.min(loss))
    #         rmseArrAvg.append(np.average(rmse_train))
    #     print("\n")
    #     for l in learningRate:
    #         print("*********************************************************************************")
    #         print("learning rate = ",l)
    #         print("*********************************************************************************")
    #         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
    #         rmse_train, loss = evaluate_algorithmGradDecent(breastcancer.values, 10, 1000, l, 0.004)
    #         lossArr2.append(loss)
    #         rmseArr2.append(rmse_train)
    #         lossArrMin2.append(np.min(loss))
    #         rmseArrAvg2.append(np.average(rmse_train))
    #
    #     print("\n\n")

    # print("Pima Indian Diabetes data: ")
    # diabetes = pd.read_csv('diabetes.csv', sep=',', header=None)
    # diabetes.reindex(np.random.permutation(diabetes.index))
    # diabetes_train_yacht_GradDecent = evaluate_algorithmGradDecent(diabetes.values, 10, 1000, 0.00001, 0.004)

    #     for t in tolerance:
    #         print("*********************************************************************************")
    #         print("tolerance = ",t)
    #         print("*********************************************************************************")
    #         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
    #         rmse_train, loss = evaluate_algorithmGradDecent(diabetes.values, 10, 1000, 0.0004, t)
    #         lossArr.append(loss)
    #         rmseArr.append(rmse_train)
    #         lossArrMin.append(np.min(loss))
    #         rmseArrAvg.append(np.average(rmse_train))
    #     print("\n")
    #     for l in learningRate:
    #         print("*********************************************************************************")
    #         print("learning rate = ",l)
    #         print("*********************************************************************************")
    #         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
    #         rmse_train, loss = evaluate_algorithmGradDecent(diabetes.values, 10, 1000, l, 0.004)
    #         lossArr2.append(loss)
    #         rmseArr2.append(rmse_train)
    #         lossArrMin2.append(np.min(loss))
    #         rmseArrAvg2.append(np.average(rmse_train))

    #     print("\n\n")
    #
    # print("Spambase data: ")
    # spambase = pd.read_csv('spambase.csv', sep=',', header=None)
    # spambase.reindex(np.random.permutation(spambase.index))
    # spambase_train_yacht_GradDecent = evaluate_algorithmGradDecent(spambase.values, 10, 1000, 0.003, 0.006)


#     for t in tolerance:
#         print("*********************************************************************************")
#         print("tolerance = ",t)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(spambase.values, 10, 1000, 0.0004, t)
#         lossArr.append(loss)
#         rmseArr.append(rmse_train)
#         lossArrMin.append(np.min(loss))
#         rmseArrAvg.append(np.average(rmse_train))
#     print("\n")
#     for l in learningRate:
#         print("*********************************************************************************")
#         print("learning rate = ",l)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(spambase.values, 10, 1000, l, 0.004)
#         lossArr2.append(loss)
#         rmseArr2.append(rmse_train)
#         lossArrMin2.append(np.min(loss))
#         rmseArrAvg2.append(np.average(rmse_train))

#     tolID = 0
#     fig1 = plt.figure()
#     ax = fig1.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : RMSE VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('RMSE')
#     for t in tolerance:
#         ax.plot(np.full(len(rmseArr[tolID]), t), rmseArr[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='RMSE VS Tolerance')
#         tolID = tolID + 1
#     tolID = 0
#     fig2 = plt.figure()
#     ax = fig2.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : Avg. RMSE VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('Avg. RMSE')
#     ax.plot(tolerance, rmseArrAvg, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. RMSE VS Tolerance')
#
#     tolID = 0
#     fig3 = plt.figure()
#     ax = fig3.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : RMSE VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('RMSE')
#     for t in learningRate:
#         ax.plot(np.full(len(rmseArr2[tolID]), t), rmseArr2[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='RMSE VS learningRate')
#         tolID = tolID + 1
#     tolID = 0
#     fig4 = plt.figure()
#     ax = fig4.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : Avg. RMSE VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('Avg. RMSE')
#     ax.plot(learningRate, rmseArrAvg2, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. RMSE VS learningRate')
#
#
#     tolID = 0
#     fig5 = plt.figure()
#     ax = fig5.add_subplot(111)
#     ax.set_title('Deliverable 1.3.2 : Loss VS Iterations by varying tolerance')
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Loss')
#     for t in tolerance:
#         ax.plot(np.arange(1, len(lossArr[tolID])+1, 1), lossArr[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS Iterations by varying tolerance')
#         tolID = tolID + 1
#     tolID = 0
#     fig6 = plt.figure()
#     ax = fig6.add_subplot(111)
#     ax.set_title('Deliverable 1.3.2 : Loss VS Iterations by varying learningRate')
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Loss')
#     for t in learningRate:
#         ax.plot(np.arange(1, len(lossArr2[tolID])+1, 1), lossArr2[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS Iterations by varying learningRate')
#         tolID = tolID + 1
#
#
#     tolID = 0
#     fig7 = plt.figure()
#     ax = fig7.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Loss VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('Loss')
#     for t in tolerance:
#         ax.plot(np.full(len(lossArr[tolID]), t), lossArr[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS Tolerance')
#         tolID = tolID + 1
#     tolID = 0
#     fig8 = plt.figure()
#     ax = fig8.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Min. Loss VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('Min. Loss')
#     ax.plot(tolerance, lossArrMin, ls='--', marker=markers[tolID], c=color[tolID], label='Min. Loss VS Tolerance')
#
#     tolID = 0
#     fig9 = plt.figure()
#     ax = fig9.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Loss VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('Loss')
#     for t in learningRate:
#         ax.plot(np.full(len(lossArr2[tolID]), t), lossArr2[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS learningRate')
#         tolID = tolID + 1
#     tolID = 0
#     fig10 = plt.figure()
#     ax = fig10.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Min. Loss VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('Min. Loss')
#     ax.plot(learningRate, lossArrMin2, ls='--', marker=markers[tolID], c=color[tolID], label='Min. Loss VS learningRate')
#
#     plt.show()

if __name__ == "__main__": main()