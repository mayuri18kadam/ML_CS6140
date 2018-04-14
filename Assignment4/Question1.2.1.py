'''
Created on Feb 8, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from math import log1p
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from scipy.special import expit
# import decimal
# decimal.getcontext().prec = 100
import matplotlib.pyplot as plt
import threading
import time

accArrAvg = []
dispLearnRate = []
exitFlag = 0
threadLock = threading.Lock()
threads = []

class myThread (threading.Thread):
    def __init__(self, threadID, counter, dataset, nfold, maxIter, learningRate):
       threading.Thread.__init__(self)
       self.threadID = threadID
       self.counter = counter
       self.dataset = dataset
       self.nfold = nfold
       self.maxIter = maxIter
       self.learningRate = learningRate
       self.acc_test = []
    def run(self):
        # dataset, n_folds, max_iter, learning_rate, threadID, counter
        self.acc_test = evaluate_algorithmGradDecent(self.dataset, self.nfold, self.maxIter, self.learningRate, self.threadID, self.counter)
        threadLock.acquire()
        accArrAvg.append(np.average(self.acc_test))
        dispLearnRate.append(self.learningRate)
        threadLock.release()

def getWeightsGradDecent(X, y, w, w_new, max_iter, learning_rate):
    for iterCount in range(max_iter):
        for rowIndex in range(len(X)):
            y_new = np.sum(np.dot(X[rowIndex], w))
            y_new = 1 if y_new > 0 else -1
            if (y[rowIndex] * y_new) <= 0:
                X_temp = []
                for ls in X[rowIndex]:
                    X_temp.append([ls])
                w_new = np.add(w_new, (learning_rate*y[rowIndex]*np.array(X_temp)))
        if np.array_equal(w, w_new) :
            return w
        np.copyto(w, w_new)
    return w


def getPredictedOutputAndWeightsGradDecent(X, y, max_iter, learning_rate):
    # appending one's array
    # X_final is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols + 1), with 1st column of 1's
    X_final = np.append(np.ones((len(X), 1)), X, axis=1)
    # bringing y in shape[[],[],...,[]] i.e. n*1
    y_final = []
    for ls in y:
        y_final.append([ls])
        # initializing weights array w[[0], [0],...[0]] to size (orig_no_of_feature_cols + 1)*1
    w = np.zeros((len(X_final[0]), 1))
    w_new = np.zeros((len(X_final[0]), 1))
    y_pred = []
    w = getWeightsGradDecent(X_final, y, w, w_new, max_iter, learning_rate)
    for rowIndex in range(len(X_final)):
        y_new = np.sum(np.dot(X_final[rowIndex], w))
        y_new = 1 if y_new > 0 else -1
        y_pred.append(y_new)
    return y_pred, w

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
    return dataset

def evaluate_algorithmGradDecent(dataset, n_folds, max_iter, learning_rate, threadID, counter):

    for i in range(counter):
        if exitFlag:
            threadID.exit()
        time.sleep(i)

    meanAcc_train = []
    meanRecall_train = []
    meanPre_train = []
    meanAcc_test = []
    meanRecall_test = []
    meanPre_test = []
    kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(dataset):
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        # dataset_train_zscore is of shape[[],[],...[]] i.e. n*(orig_no_of_all_cols)
        meanArr = np.mean(dataset_train, axis=0)
        stdArr = np.std(dataset_train, axis=0)
        dataset_train_zscore = calZScore(dataset_train, meanArr, stdArr)
        # print("dataset_train_zscore = ",dataset_train_zscore)
        dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)

        # X is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols)
        X_train, X_test = dataset_train_zscore[:, 0:-1], dataset_test_zscore[:, 0:-1]
        y_train, y_test = dataset_train[:, -1], dataset_test[:, -1]

        y_pred_train, w_train = getPredictedOutputAndWeightsGradDecent(X_train, y_train, max_iter, learning_rate)

        X_test_final = np.append(np.ones((len(X_test), 1)), X_test, axis=1)
        y_pred_test = np.matmul(X_test_final, w_train)
        y_pred_test_final = []
        for y_data in y_pred_test:
            y_pred_test_final.append(-1) if y_data < 0 else y_pred_test_final.append(1)

        meanAcc_train.append(accuracy_score(y_train, y_pred_train))
        meanRecall_train.append(recall_score(y_train, y_pred_train, average='micro'))
        meanPre_train.append(precision_score(y_train, y_pred_train, average='micro'))
        print("confusion_matrix(y_train, y_pred_train_final) = \n", confusion_matrix(y_train, y_pred_train))

        meanAcc_test.append(accuracy_score(y_test, y_pred_test_final))
        meanRecall_test.append(recall_score(y_test, y_pred_test_final, average='micro'))
        meanPre_test.append(precision_score(y_test, y_pred_test_final, average='micro'))
        print("confusion_matrix(y_test, y_pred_test_final) = \n", confusion_matrix(y_test, y_pred_test_final))
        print("-------------------------------------------------------------------------------------------------")

    print("*********************************************************************************")
    print("learning rate = ", learning_rate)
    print("*********************************************************************************")

    print("Train Accuracy = ", meanAcc_train)
    print("Train Accuracy Mean = ", np.average(meanAcc_train))
    print("Train Accuracy standard deviation = ", np.std(meanAcc_train))
    print("Train Recall = ", meanRecall_train)
    print("Train Recall Mean = ", np.average(meanRecall_train))
    print("Train Recall standard deviation = ", np.std(meanRecall_train))
    print("Train Precision = ", meanPre_train)
    print("Train Precision Mean = ", np.average(meanPre_train))
    print("Train Precision standard deviation = ", np.std(meanPre_train))
    print("")

    print("Test Accuracy = ", meanAcc_test)
    print("Test Accuracy Mean = ", np.average(meanAcc_test))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_test))
    print("Test Recall = ", meanRecall_test)
    print("Test Recall Mean = ", np.average(meanRecall_test))
    print("Test Recall standard deviation = ", np.std(meanRecall_test))
    print("Test Precision = ", meanPre_test)
    print("Test Precision Mean = ", np.average(meanPre_test))
    print("Test Precision standard deviation = ", np.std(meanPre_test))

    return meanAcc_test

def main():
    color = ['y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm', 'c', 'r', 'g', 'b', 'k']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'd', 'D']
    learningRate = np.array([0.00001, 0.00005, 0.00009, 0.0004, 0.0008, 0.003, 0.007, 0.01, 0.05, 0.09])

    print("perceptronData: ")
    perceptronData = pd.read_csv('perceptronData.csv', sep=',', header=None)
    perceptronData.reindex(np.random.permutation(perceptronData.index))
    # self, threadID, counter, dataset, nfold, maxIter, learningRate
    thread1 = myThread(1, 1, perceptronData.values, 10, 1000, 0.00001)
    thread1.start()
    threads.append(thread1)

    # i = 0
    # for l in learningRate:
    #     # self, threadID, counter, dataset, nfold, maxIter, learningRate
    #     thread1 = myThread(i, i, perceptronData.values, 10, 1000, l)
    #     i = i+1
    #     thread1.start()
    #     threads.append(thread1)

    for t in threads:
        t.join()
    print("Exiting Main Thread")

    # tolID = 0
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111)
    # ax.set_title('Avg. Accuracy VS learningRate')
    # ax.set_xlabel('learningRate')
    # ax.set_ylabel('Avg. Accuracy')
    # ax.plot(dispLearnRate, accArrAvg, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. Accuracy VS learningRate')
    # plt.show()

if __name__ == "__main__": main()