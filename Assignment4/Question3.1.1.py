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
# import threading
# import time
from sklearn import svm
from multiprocessing import Process

meanAcc_testSVM = []
meanRecall_testSVM = []
meanPre_testSVM = []
meanAcc_testRBF = []
meanRecall_testRBF = []
meanPre_testRBF = []

bestFinalCArr = []
bestFinalSVMAccArr = []
bestFinalGammaArr = []
bestFinalRBFAccArr = []

exitFlag = 0
processes = []

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
    return dataset

def getBestParams(dataset_train2, dataset_test2, c, gamma, bestOuterC, bestOuterSVMAcc, bestOuterGamma, bestOuterRBFAcc) :
    bestC = 0
    bestSVMAcc = 0
    for c1 in c:
        modelSVM = svm.SVC(kernel='linear', C=c1)
        modelSVM.fit(dataset_train2[:, 0:-1], dataset_train2[:, -1])
        y_pred_test_1 = modelSVM.predict(dataset_test2[:, 0:-1])
        if (accuracy_score(dataset_test2[:, -1], y_pred_test_1) > bestSVMAcc):
            bestSVMAcc = accuracy_score(dataset_test2[:, -1], y_pred_test_1)
            bestC = c1
    if (bestSVMAcc > bestOuterSVMAcc):
        bestOuterSVMAcc = bestSVMAcc
        bestOuterC = bestC

    bestGamma = 0
    bestRBFAcc = 0
    for g in gamma:
        modelRBF = svm.SVC(kernel='rbf', gamma=g, C=bestOuterC)
        modelRBF.fit(dataset_train2[:, 0:-1], dataset_train2[:, -1])
        y_pred_test_2 = modelRBF.predict(dataset_test2[:, 0:-1])
        if (accuracy_score(dataset_test2[:, -1], y_pred_test_2) > bestRBFAcc):
            bestRBFAcc = accuracy_score(dataset_test2[:, -1], y_pred_test_2)
            bestGamma = g
    if (bestRBFAcc > bestOuterRBFAcc):
        bestOuterRBFAcc = bestRBFAcc
        bestOuterGamma = bestGamma

    return bestOuterC, bestOuterSVMAcc, bestOuterGamma, bestOuterRBFAcc

def getStats(dataset_train_zscore, dataset_test_zscore, mfold, c, gamma):
    bestOuterC = 0
    bestOuterSVMAcc = 0
    bestOuterGamma = 0
    bestOuterRBFAcc = 0
    kf2 = KFold(n_splits=mfold, random_state=None, shuffle=True)
    for train_index2, test_index2 in kf2.split(dataset_train_zscore):
        dataset_train2, dataset_test2 = dataset_train_zscore[train_index2], dataset_train_zscore[test_index2]

        # dataset_train2, dataset_test2, c, gamma, bestOuterC, bestOuterSVMAcc, bestOuterGamma, bestOuterRBFAcc
        bestOuterC, bestOuterSVMAcc, bestOuterGamma, bestOuterRBFAcc = getBestParams(dataset_train2, dataset_test2,
                                                                                     c, gamma, bestOuterC,
                                                                                     bestOuterSVMAcc, bestOuterGamma,
                                                                                     bestOuterRBFAcc)

    # print("bestOuterC = ",bestOuterC)
    # print("bestOuterSVMAcc = ", bestOuterSVMAcc)
    # print("bestOuterGamma = ", bestOuterGamma)
    # print("bestOuterRBFAcc = ", bestOuterRBFAcc)
    # print("*********************************************************************************************")
    modelSVM = svm.SVC(kernel='linear', C=bestOuterC)
    modelSVM.fit(dataset_train_zscore[:, 0:-1], dataset_train_zscore[:, -1])
    y_pred_test_1 = modelSVM.predict(dataset_test_zscore[:, 0:-1])
    accAcoreSVM = accuracy_score(dataset_test_zscore[:, -1], y_pred_test_1)
    recScoreSVM = recall_score(dataset_test_zscore[:, -1], y_pred_test_1, average='micro')
    preScoreSVM = precision_score(dataset_test_zscore[:, -1], y_pred_test_1, average='micro')

    modelRBF = svm.SVC(kernel='rbf', gamma=bestOuterGamma, C=bestOuterC)
    modelRBF.fit(dataset_train_zscore[:, 0:-1], dataset_train_zscore[:, -1])
    y_pred_test_2 = modelRBF.predict(dataset_test_zscore[:, 0:-1])
    accAcoreRBF = accuracy_score(dataset_test_zscore[:, -1], y_pred_test_2)
    recScoreRBF = recall_score(dataset_test_zscore[:, -1], y_pred_test_2, average='micro')
    preScoreRBF = precision_score(dataset_test_zscore[:, -1], y_pred_test_2, average='micro')

    bestFinalCArr.append(bestOuterC)
    bestFinalSVMAccArr.append(bestOuterSVMAcc)
    bestFinalGammaArr.append(bestOuterGamma)
    bestFinalRBFAccArr.append(bestOuterRBFAcc)

    meanAcc_testSVM.append(accAcoreSVM)
    meanRecall_testSVM.append(recScoreSVM)
    meanPre_testSVM.append(preScoreSVM)
    meanAcc_testRBF.append(accAcoreRBF)
    meanRecall_testRBF.append(recScoreRBF)
    meanPre_testRBF.append(preScoreRBF)

def evaluate_algorithm(dataset, n_folds, mfold, c, gamma):

    kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(dataset):
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        # dataset_train_zscore is of shape[[],[],...[]] i.e. n*(orig_no_of_all_cols)
        meanArr = np.mean(dataset_train, axis=0)
        stdArr = np.std(dataset_train, axis=0)
        dataset_train_zscore = calZScore(dataset_train, meanArr, stdArr)
        dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)

        getStats(dataset_train_zscore, dataset_test_zscore, mfold, c, gamma)

    indxMaxC = np.argmax(bestFinalSVMAccArr)
    indxMaxGamma = np.argmax(bestFinalRBFAccArr)

    print("*********************************************************************************")
    print("best C = ", bestFinalCArr[indxMaxC])
    print("best Gamma = ", bestFinalGammaArr[indxMaxGamma])
    print("*********************************************************************************")
    print("SVM = ")
    print("Avg. Accuracy = ", meanAcc_testSVM)
    print("Test Accuracy Mean = ", np.average(meanAcc_testSVM))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_testSVM))
    print("Test Recall Mean = ", np.average(meanRecall_testSVM))
    print("Test Recall standard deviation = ", np.std(meanRecall_testSVM))
    print("Test Precision Mean = ", np.average(meanPre_testSVM))
    print("Test Precision standard deviation = ", np.std(meanPre_testSVM))

    print("*********************************************************************************")
    print("RBF = ")
    print("Avg. Accuracy = ", meanAcc_testRBF)
    print("Test Accuracy Mean = ", np.average(meanAcc_testRBF))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_testRBF))
    print("Test Recall Mean = ", np.average(meanRecall_testRBF))
    print("Test Recall standard deviation = ", np.std(meanRecall_testRBF))
    print("Test Precision Mean = ", np.average(meanPre_testRBF))
    print("Test Precision standard deviation = ", np.std(meanPre_testRBF))
    print("*********************************************************************************")

def main():
    color = ['y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'd', 'D', 'o', 'v']
    c = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    gamma = np.array([0.00003051757, 0.00006103515, 0.00012207031, 0.00024414062, 0.00048828125, 0.0009765625,
                      0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])

    # print("Breast Cancer data: ")
    # breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    # breastcancer.reindex(np.random.permutation(breastcancer.index))
    # self, threadID, counter, dataset, nfold, mfold, c, gamma
    # evaluate_algorithm(breastcancer.values, 10, 5, c, gamma)

    print("Pima Indian Diabetes data: ")
    diabetes = pd.read_csv('diabetes.csv', sep=',', header=None)
    diabetes.reindex(np.random.permutation(diabetes.index))
    # # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    evaluate_algorithm(diabetes.values, 10, 5, c, gamma)

    # print("Spambase data: ")
    # spambase = pd.read_csv('spambase.csv', sep=',', header=None)
    # spambase.reindex(np.random.permutation(spambase.index))
    # # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    # evaluate_algorithm(spambase.values, 10, 5, c, gamma)

if __name__ == "__main__": main()