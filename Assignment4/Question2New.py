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
import matplotlib.pyplot as plt
import threading
import time

accArrAvg = []
lossArrMin = []
dispTol = []
dispLearnRate = []
displambda = []
exitFlag = 0
threadLock = threading.Lock()
threads = []

class myThread (threading.Thread):
    def __init__(self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam):
       threading.Thread.__init__(self)
       self.threadID = threadID
       self.counter = counter
       self.dataset = dataset
       self.nfold = nfold
       self.maxIter = maxIter
       self.learningRate = learningRate
       self.tol = tol
       self.lam = lam
       self.acc_test = []
       self.loss = []
    def run(self):
        acc_test, loss = evaluate_algorithmGradDecent(self.dataset, self.nfold, self.maxIter, self.learningRate,
                                    self.tol, self.lam, self.threadID, self.counter)
        threadLock.acquire()
        lossArrMin.append(np.min(loss))
        accArrAvg.append(np.average(acc_test))
        dispTol.append(self.tol)
        dispLearnRate.append(self.learningRate)
        displambda.append(self.lam)
        threadLock.release()

def calLossFunc(X, y, w):
    loss = 0
    sigArr = []
    for rowIndex in range(len(X)):
        sig = expit(np.matmul(X[rowIndex], w))
        sigArr.append(sig)
        loss = loss - (y[rowIndex] * np.array(log1p(sig)) + (np.array(1) - y[rowIndex]) * np.array(log1p(1 - sig)))
    return loss, sigArr

def getWeightsGradDecent(X, y, w, max_iter, learning_rate, tolerance, lam):
    loss = 0
    lossArr = []
    for iterCount in range(max_iter):
        loss_new, sigArr = calLossFunc(X, y, w)
        lossArr.append(loss_new)
        if abs(loss_new - loss) <= tolerance:
            return w, lossArr
        updated = []
        for rowIndex in range(len(X)):
            updated.append(sigArr[rowIndex] - y[rowIndex])
        # print("w = ",w)
        # print("(lam/2 * (np.dot(w,w))) = ",(lam/2)*(np.matmul(w.T,w)))
        # w = w - learning_rate * ((np.matmul(np.transpose(X), updated)) + ((lam/2) * np.sum(np.dot(w,2))))
        w = w - learning_rate * ((np.matmul(np.transpose(X), updated)) + (lam/2)*(np.matmul(w.T,w)))
        loss = loss_new
    return w, lossArr

def getPredictedOutputAndWeightsGradDecent(X, y, max_iter, learning_rate, tolerance, lam):
    # appending one's array
    # X_final is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols + 1), with 1st column of 1's
    X_final = np.append(np.ones((len(X), 1)), X, axis=1)
    # bringing y in shape[[],[],...,[]] i.e. n*1
    y_final = []
    for ls in y:
        y_final.append([ls])
        # initializing weights array w[[0], [0],...[0]] to size (orig_no_of_feature_cols + 1)*1
    w = np.zeros((len(X_final[0]), 1))
    w, lossArr = getWeightsGradDecent(X_final, y_final, w, max_iter, learning_rate, tolerance, lam)
    y_pred = []
    for rowIndex in range(len(X)):
        y_pred.append(expit(np.matmul(X_final[rowIndex], w)))
    return y_pred, w, lossArr

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
    return dataset

def evaluate_algorithmGradDecent(dataset, n_folds, max_iter, learning_rate, tolerance, lam, threadID, counter):

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

        y_pred_train, w_train, lossArr = getPredictedOutputAndWeightsGradDecent(X_train, y_train, max_iter,
                                                                                learning_rate, tolerance, lam)
        y_pred_train_final = []
        for y_data in y_pred_train:
            y_pred_train_final.append(0) if y_data < 0.5 else y_pred_train_final.append(1)

        X_test_final = np.append(np.ones((len(X_test), 1)), X_test, axis=1)
        y_pred_test = np.matmul(X_test_final, w_train)
        y_pred_test_final = []
        for y_data in y_pred_test:
            y_pred_test_final.append(0) if y_data < 0.5 else y_pred_test_final.append(1)

        meanAcc_train.append(accuracy_score(y_train, y_pred_train_final))
        meanRecall_train.append(recall_score(y_train, y_pred_train_final, average='micro'))
        meanPre_train.append(precision_score(y_train, y_pred_train_final, average='micro'))
        # print("confusion_matrix(y_train, y_pred_train_final) = \n", confusion_matrix(y_train, y_pred_train_final))

        meanAcc_test.append(accuracy_score(y_test, y_pred_test_final))
        meanRecall_test.append(recall_score(y_test, y_pred_test_final, average='micro'))
        meanPre_test.append(precision_score(y_test, y_pred_test_final, average='micro'))
        # print("confusion_matrix(y_test, y_pred_test_final) = \n", confusion_matrix(y_test, y_pred_test_final))
        # print("-------------------------------------------------------------------------------------------------")

    print("*********************************************************************************")
    print("tolerance = ", tolerance)
    print("learning rate = ", learning_rate)
    print("lambda = ", lam)
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

    return meanAcc_test, lossArr

def main():
    color = ['y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm', 'c', 'r', 'g', 'b', 'k']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'd', 'D']
    tolerance = np.array([0.0005, 0.005, 0.05,])
    learningRate = np.array([0.00005, 0.0005, 0.005])
    lam = np.array([0.05, 0.5, 5])

    print("Breast Cancer data: ")
    breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    breastcancer.reindex(np.random.permutation(breastcancer.index))
    # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    thread1 = myThread(1, 1, breastcancer.values, 10, 1000, 0.005, 0.0005, 0.5)
    thread1.start()
    threads.append(thread1)

    # i = 0
    # for t in tolerance:
    #     for l in learningRate:
    #         for lamVal in lam:
    #             # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    #             thread1 = myThread(1, i, breastcancer.values, 10, 1000, l, t, lamVal)
    #             i = i+1
    #             thread1.start()
    #             threads.append(thread1)

    print("Pima Indian Diabetes data: ")
    diabetes = pd.read_csv('diabetes.csv', sep=',', header=None)
    diabetes.reindex(np.random.permutation(diabetes.index))
    # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    thread2 = myThread(2, 2, diabetes.values, 10, 1000, 0.005, 0.005, 5)
    thread2.start()
    threads.append(thread2)

    # i = 0
    # for t in tolerance:
    #     for l in learningRate:
    #         for lamVal in lam:
    #             # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    #             thread1 = myThread(1, i, diabetes.values, 10, 1000, l, t, lamVal)
    #             i = i + 1
    #             thread1.start()
    #             threads.append(thread1)

    print("Spambase data: ")
    spambase = pd.read_csv('spambase.csv', sep=',', header=None)
    spambase.reindex(np.random.permutation(spambase.index))
    # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    thread3 = myThread(3, 3, spambase.values, 10, 1000, 0.00005, 0.005, 5)
    thread3.start()
    threads.append(thread3)

    # i = 0
    # for t in tolerance:
    #     for l in learningRate:
    #         for lamVal in lam:
    #             # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    #             thread1 = myThread(1, i, spambase.values, 10, 1000, l, t, lamVal)
    #             i = i + 1
    #             thread1.start()
    #             threads.append(thread1)

    for t in threads:
        t.join()
    print("Exiting Main Thread")

    # tolID = 0
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111)
    # ax.set_title('Avg. Accuracy VS Tolerance')
    # ax.set_xlabel('Tolerance')
    # ax.set_ylabel('Avg. Accuracy')
    # ax.plot(dispTol, accArrAvg, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. Accuracy VS Tolerance')
    #
    # tolID = 1
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(111)
    # ax.set_title('Avg. Accuracy VS learningRate')
    # ax.set_xlabel('learningRate')
    # ax.set_ylabel('Avg. Accuracy')
    # ax.plot(dispLearnRate, accArrAvg, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. Accuracy VS learningRate')
    #
    # tolID = 2
    # fig3 = plt.figure()
    # ax = fig3.add_subplot(111)
    # ax.set_title('Avg. Accuracy VS Lambda')
    # ax.set_xlabel('Lambda')
    # ax.set_ylabel('Avg. Accuracy')
    # ax.plot(displambda, accArrAvg, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. Accuracy VS Lambda')
    #
    # tolID = 3
    # fig4 = plt.figure()
    # ax = fig4.add_subplot(111)
    # ax.set_title('Min. Loss VS Tolerance')
    # ax.set_xlabel('Tolerance')
    # ax.set_ylabel('Min. Loss')
    # ax.plot(dispTol, lossArrMin, ls='--', marker=markers[tolID], c=color[tolID], label='Min. Loss VS Tolerance')
    #
    # tolID = 4
    # fig5 = plt.figure()
    # ax = fig5.add_subplot(111)
    # ax.set_title('Min. Loss VS learningRate')
    # ax.set_xlabel('learningRate')
    # ax.set_ylabel('Min. Loss')
    # ax.plot(dispLearnRate, lossArrMin, ls='--', marker=markers[tolID], c=color[tolID], label='Min. Loss VS learningRate')
    #
    # tolID = 5
    # fig6 = plt.figure()
    # ax = fig6.add_subplot(111)
    # ax.set_title('Min. Loss VS Lambda')
    # ax.set_xlabel('Lambda')
    # ax.set_ylabel('Min. Loss')
    # ax.plot(displambda, lossArrMin, ls='--', marker=markers[tolID], c=color[tolID], label='Min. Loss VS Lambda')
    #
    # plt.show()

if __name__ == "__main__": main()