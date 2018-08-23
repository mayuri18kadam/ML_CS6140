'''
Created on Feb 8, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import svm
import matplotlib.pyplot as plt

meanAcc_testSVM = []
meanAcc_testRBF = []
bestFinalCArr = []
bestFinalSVMAccArr = []
bestFinalGammaArr = []
bestFinalRBFAccArr = []

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
        if (roc_auc_score(dataset_test2[:, -1], y_pred_test_1) > bestSVMAcc):
            bestSVMAcc = roc_auc_score(dataset_test2[:, -1], y_pred_test_1)
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
        if (roc_auc_score(dataset_test2[:, -1], y_pred_test_2) > bestRBFAcc):
            bestRBFAcc = roc_auc_score(dataset_test2[:, -1], y_pred_test_2)
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
    modelSVM = svm.SVC(kernel='linear', C=bestOuterC)
    modelSVM.fit(dataset_train_zscore[:, 0:-1], dataset_train_zscore[:, -1])
    y_pred_test_1 = modelSVM.predict(dataset_test_zscore[:, 0:-1])
    accAcoreSVM = roc_auc_score(dataset_test_zscore[:, -1], y_pred_test_1)

    modelRBF = svm.SVC(kernel='rbf', gamma=bestOuterGamma, C=bestOuterC)
    modelRBF.fit(dataset_train_zscore[:, 0:-1], dataset_train_zscore[:, -1])
    y_pred_test_2 = modelRBF.predict(dataset_test_zscore[:, 0:-1])
    accAcoreRBF = roc_auc_score(dataset_test_zscore[:, -1], y_pred_test_2)

    bestFinalCArr.append(bestOuterC)
    bestFinalSVMAccArr.append(bestOuterSVMAcc)
    bestFinalGammaArr.append(bestOuterGamma)
    bestFinalRBFAccArr.append(bestOuterRBFAcc)

    meanAcc_testSVM.append(accAcoreSVM)
    meanAcc_testRBF.append(accAcoreRBF)

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
    print("Train ROC-AUC = ", meanAcc_testSVM)
    print("Test Accuracy Mean = ", np.average(meanAcc_testSVM))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_testSVM))

    print("*********************************************************************************")
    print("RBF = ")
    print("Test ROC-AUC = ", meanAcc_testRBF)
    print("Test Accuracy Mean = ", np.average(meanAcc_testRBF))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_testRBF))
    print("*********************************************************************************")

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_title('Linear kernel - ROC-AUC VS n-Folds')
    ax.set_xlabel('n-Folds')
    ax.set_ylabel('ROC-AUC')
    ax.plot(np.arange(1, 11, 1), meanAcc_testSVM, ls='--', marker='^', c='g', label='ROC-AUC VS n-Folds')

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.set_title('RBF kernel - ROC-AUC VS n-Folds')
    ax.set_xlabel('n-Folds')
    ax.set_ylabel('ROC-AUC')
    ax.plot(np.arange(1, 11, 1), meanAcc_testRBF, ls='--', marker='<', c='b', label='ROC-AUC VS n-Folds')

    plt.show()


def main():
    color = ['y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'd', 'D', 'o', 'v']
    c = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    gamma = np.array([0.00003051757, 0.00006103515, 0.00012207031, 0.00024414062, 0.00048828125, 0.0009765625,
                      0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])

    print("Breast Cancer data: ")
    breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    breastcancer.reindex(np.random.permutation(breastcancer.index))
    # self, threadID, counter, dataset, nfold, mfold, c, gamma
    evaluate_algorithm(breastcancer.values, 10, 5, c, gamma)

    # print("Pima Indian Diabetes data: ")
    # diabetes = pd.read_csv('diabetes.csv', sep=',', header=None)
    # diabetes.reindex(np.random.permutation(diabetes.index))
    # # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    # evaluate_algorithm(diabetes.values, 10, 5, c, gamma)

    # print("Spambase data: ")
    # spambase = pd.read_csv('spambase.csv', sep=',', header=None)
    # spambase.reindex(np.random.permutation(spambase.index))
    # # self, threadID, counter, dataset, nfold, maxIter, learningRate, tol, lam
    # evaluate_algorithm(spambase.values, 10, 5, c, gamma)

if __name__ == "__main__": main()