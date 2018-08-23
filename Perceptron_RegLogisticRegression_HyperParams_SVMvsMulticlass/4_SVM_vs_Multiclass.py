'''
Created on Feb 8, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import svm
import matplotlib.pyplot as plt

# meanAcc_testSVM = []
# meanAcc_testRBF = []
#
# meanAUC_testSVM = []
# meanAUC_testRBF = []
#
# meanRecall_testSVM = []
# meanPre_testSVM = []
# meanRecall_testRBF = []
# meanPre_testRBF = []
#
# bestFinalCArr = []
# bestFinalSVMAccArr = []
# bestFinalGammaArr = []
# bestFinalRBFAccArr = []

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
    modelSVM = svm.SVC(kernel='linear', C=bestOuterC)
    modelSVM.fit(dataset_train_zscore[:, 0:-1], dataset_train_zscore[:, -1])
    y_pred_test_1 = modelSVM.predict(dataset_test_zscore[:, 0:-1])
    accAcoreSVM = accuracy_score(dataset_test_zscore[:, -1], y_pred_test_1)
    recScoreSVM = recall_score(dataset_test_zscore[:, -1], y_pred_test_1, average='micro')
    preScoreSVM = precision_score(dataset_test_zscore[:, -1], y_pred_test_1, average='micro')
    aucScoreSVM = roc_auc_score(dataset_test_zscore[:, -1], y_pred_test_1)

    modelRBF = svm.SVC(kernel='rbf', gamma=bestOuterGamma, C=bestOuterC)
    modelRBF.fit(dataset_train_zscore[:, 0:-1], dataset_train_zscore[:, -1])
    y_pred_test_2 = modelRBF.predict(dataset_test_zscore[:, 0:-1])
    accAcoreRBF = accuracy_score(dataset_test_zscore[:, -1], y_pred_test_2)
    recScoreRBF = recall_score(dataset_test_zscore[:, -1], y_pred_test_2, average='micro')
    preScoreRBF = precision_score(dataset_test_zscore[:, -1], y_pred_test_2, average='micro')
    aucScoreRBF = roc_auc_score(dataset_test_zscore[:, -1], y_pred_test_2)

    return bestOuterC, bestOuterSVMAcc, bestOuterGamma, bestOuterRBFAcc, \
           accAcoreSVM, accAcoreRBF, recScoreSVM, recScoreRBF, preScoreSVM, preScoreRBF, aucScoreSVM, aucScoreRBF

def evaluate_algorithm(dataset, n_folds, mfold, c, gamma):
    meanAcc_testSVM = []
    meanAcc_testRBF = []
    meanAUC_testSVM = []
    meanAUC_testRBF = []
    meanRecall_testSVM = []
    meanPre_testSVM = []
    meanRecall_testRBF = []
    meanPre_testRBF = []
    bestFinalCArr = []
    bestFinalSVMAccArr = []
    bestFinalGammaArr = []
    bestFinalRBFAccArr = []

    kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(dataset):
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        # dataset_train_zscore is of shape[[],[],...[]] i.e. n*(orig_no_of_all_cols)
        meanArr = np.mean(dataset_train, axis=0)
        stdArr = np.std(dataset_train, axis=0)
        dataset_train_zscore = calZScore(dataset_train, meanArr, stdArr)
        dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)

        bestOuterC, bestOuterSVMAcc, bestOuterGamma, bestOuterRBFAcc, \
        accAcoreSVM, accAcoreRBF, recScoreSVM, recScoreRBF, preScoreSVM, \
        preScoreRBF, aucScoreSVM, aucScoreRBF = getStats(dataset_train_zscore, dataset_test_zscore, mfold, c, gamma)

        bestFinalCArr.append(bestOuterC)
        bestFinalSVMAccArr.append(bestOuterSVMAcc)
        bestFinalGammaArr.append(bestOuterGamma)
        bestFinalRBFAccArr.append(bestOuterRBFAcc)

        meanAcc_testSVM.append(accAcoreSVM)
        meanAcc_testRBF.append(accAcoreRBF)
        meanAUC_testSVM.append(aucScoreSVM)
        meanAUC_testRBF.append(aucScoreRBF)
        meanRecall_testSVM.append(recScoreSVM)
        meanPre_testSVM.append(preScoreSVM)
        meanRecall_testRBF.append(recScoreRBF)
        meanPre_testRBF.append(preScoreRBF)

    indxMaxC = np.argmax(bestFinalSVMAccArr)
    indxMaxGamma = np.argmax(bestFinalRBFAccArr)

    print("*********************************************************************************")
    print("best C = ", bestFinalCArr[indxMaxC])
    print("best Gamma = ", bestFinalGammaArr[indxMaxGamma])
    print("*********************************************************************************")
    print("SVM = ")
    print("Test Accuracy Mean = ", np.average(meanAcc_testSVM))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_testSVM))
    print("Test AUC-ROC Mean = ", np.average(meanAUC_testSVM))
    print("Test AUC-ROC standard deviation = ", np.std(meanAUC_testSVM))
    print("Test Recall Mean = ", np.average(meanRecall_testSVM))
    print("Test Recall standard deviation = ", np.std(meanRecall_testSVM))
    print("Test Precision Mean = ", np.average(meanPre_testSVM))
    print("Test Precision standard deviation = ", np.std(meanPre_testSVM))
    print("*********************************************************************************")
    print("RBF = ")
    print("Test Accuracy Mean = ", np.average(meanAcc_testRBF))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_testRBF))
    print("Test AUC-ROC Mean = ", np.average(meanAUC_testRBF))
    print("Test AUC-ROC standard deviation = ", np.std(meanAUC_testRBF))
    print("Test Recall Mean = ", np.average(meanRecall_testRBF))
    print("Test Recall standard deviation = ", np.std(meanRecall_testRBF))
    print("Test Precision Mean = ", np.average(meanPre_testRBF))
    print("Test Precision standard deviation = ", np.std(meanPre_testRBF))
    print("*********************************************************************************")

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_title('Linear kernel - ROC-AUC VS n-Folds')
    ax.set_xlabel('n-Folds')
    ax.set_ylabel('ROC-AUC')
    ax.plot(np.arange(1, 11, 1), meanAUC_testSVM, ls='--', marker='^', c='g', label='ROC-AUC VS n-Folds')

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.set_title('RBF kernel - ROC-AUC VS n-Folds')
    ax.set_xlabel('n-Folds')
    ax.set_ylabel('ROC-AUC')
    ax.plot(np.arange(1, 11, 1), meanAUC_testRBF, ls='--', marker='<', c='b', label='ROC-AUC VS n-Folds')

    plt.show()

    return bestFinalCArr[indxMaxC], bestFinalGammaArr[indxMaxGamma]

def dataPreProcess (dataset):
    X = dataset[:, 1:]
    y = dataset[:, 0]
    y_final = []
    for ls in y:
        y_final.append([ls])
    # print(y_final)
    actualDS = np.concatenate((X, y_final), axis=1)
    y1 = np.array([1 if n == 1 else -1 for n in y])
    # print(y1)
    y_final1 = []
    for ls in y1:
        y_final1.append([ls])
    DS1 = np.concatenate((X, y_final1), axis=1)

    y2 = np.array([1 if n == 2 else -1 for n in y])
    y_final2 = []
    for ls in y2:
        y_final2.append([ls])
    DS2 = np.concatenate((X, y_final2), axis=1)

    y3 = np.array([1 if n == 3 else -1 for n in y])
    y_final3 = []
    for ls in y3:
        y_final3.append([ls])
    DS3 = np.concatenate((X, y_final3), axis=1)

    return actualDS, DS1, DS2, DS3

def computeProb(actualDS, DS1, DS2, DS3, c1, gamma1, c2, gamma2, c3, gamma3) :
    meanAcc_test = []
    meanRecall_test = []
    meanPre_test = []
    # meanAuc_test = []

    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(actualDS):
        dataset_train, dataset_test = actualDS[train_index], actualDS[test_index]
        DS1_train, DS1_test = DS1[train_index], DS1[test_index]
        DS2_train, DS2_test = DS2[train_index], DS2[test_index]
        DS3_train, DS3_test = DS3[train_index], DS3[test_index]

        meanArr = np.mean(dataset_train, axis=0)
        stdArr = np.std(dataset_train, axis=0)
        dataset_train_final = calZScore(dataset_train, meanArr, stdArr)
        dataset_test_final = calZScore(dataset_test, meanArr, stdArr)

        meanArr1 = np.mean(DS1_train, axis=0)
        stdArr1 = np.std(DS1_train, axis=0)
        DS1_train_final = calZScore(DS1_train, meanArr1, stdArr1)
        DS1_test_final = calZScore(DS1_test, meanArr1, stdArr1)

        meanArr2 = np.mean(DS2_train, axis=0)
        stdArr2 = np.std(DS2_train, axis=0)
        DS2_train_final = calZScore(DS2_train, meanArr2, stdArr2)
        DS2_test_final = calZScore(DS2_test, meanArr2, stdArr2)

        meanArr3 = np.mean(DS3_train, axis=0)
        stdArr3 = np.std(DS3_train, axis=0)
        DS3_train_final = calZScore(DS3_train, meanArr3, stdArr3)
        DS3_test_final = calZScore(DS3_test, meanArr3, stdArr3)

        modelSVM1 = svm.SVC(kernel='linear', C=c1, probability=True)
        modelSVM1.fit(DS1_train_final[:, 0:-1], DS1_train_final[:, -1])
        modelRBF1 = svm.SVC(kernel='rbf', gamma=gamma1, C=c1)
        modelRBF1.fit(DS1_train_final[:, 0:-1], DS1_train_final[:, -1])

        modelSVM2 = svm.SVC(kernel='linear', C=c2, probability=True)
        modelSVM2.fit(DS2_train_final[:, 0:-1], DS2_train_final[:, -1])
        modelRBF2 = svm.SVC(kernel='rbf', gamma=gamma2, C=c2)
        modelRBF2.fit(DS2_train_final[:, 0:-1], DS2_train_final[:, -1])

        modelSVM3 = svm.SVC(kernel='linear', C=c3, probability=True)
        modelSVM3.fit(DS3_train_final[:, 0:-1], DS3_train_final[:, -1])
        modelRBF3 = svm.SVC(kernel='rbf', gamma=gamma3, C=c3)
        modelRBF3.fit(DS3_train_final[:, 0:-1], DS3_train_final[:, -1])

        y_pred_test_1 = modelSVM1.predict_proba(DS1_test_final[:, 0:-1])
        y_pred_test_2 = modelSVM2.predict_proba(DS2_test_final[:, 0:-1])
        y_pred_test_3 = modelSVM3.predict_proba(DS3_test_final[:, 0:-1])

        # print("y_pred_test_1 = ",y_pred_test_1)
        # print("dataset_test_final[:,-1] = ",dataset_test_final[:,-1])

        y_pred = []
        i = 0
        for row in dataset_test_final[:,-1]:
            maxProb = max(y_pred_test_1[i, 1], y_pred_test_2[i, 1], y_pred_test_3[i, 1])
            if maxProb == y_pred_test_1[i, 1] :
                y_pred.append(1)
            elif maxProb == y_pred_test_2[i, 1]:
                y_pred.append(2)
            else:
                y_pred.append(3)
            i = i + 1

        accAcore = accuracy_score(dataset_test_final[:,-1], y_pred)
        recScore = recall_score(dataset_test_final[:,-1], y_pred, average='micro')
        preScore = precision_score(dataset_test_final[:,-1], y_pred, average='micro')
        # aucScore = roc_auc_score(dataset_test_final[:,-1], y_pred)
        meanAcc_test.append(accAcore)
        meanRecall_test.append(recScore)
        meanPre_test.append(preScore)
        # meanAuc_test.append(aucScore)

    print("Test Accuracy = ", meanAcc_test)
    print("Test Accuracy Mean = ", np.average(meanAcc_test))
    print("Test Accuracy standard deviation = ", np.std(meanAcc_test))
    print("Test Recall = ", meanRecall_test)
    print("Test Recall Mean = ", np.average(meanRecall_test))
    print("Test Recall standard deviation = ", np.std(meanRecall_test))
    print("Test Precision = ", meanPre_test)
    print("Test Precision Mean = ", np.average(meanPre_test))
    print("Test Precision standard deviation = ", np.std(meanPre_test))

def main():
    color = ['y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'd', 'D', 'o', 'v']
    c = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    gamma = np.array([0.00003051757, 0.00006103515, 0.00012207031, 0.00024414062, 0.00048828125, 0.0009765625,
                      0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32])

    print("wine data: ")
    wine = pd.read_csv('wine.csv', sep=',', header=None)
    wine.reindex(np.random.permutation(wine.index))
    actualDS, DS1, DS2, DS3 = dataPreProcess(wine.values)
    c1, gamma1 = evaluate_algorithm(DS1, 10, 5, c, gamma)
    c2, gamma2 = evaluate_algorithm(DS2, 10, 5, c, gamma)
    c3, gamma3 = evaluate_algorithm(DS2, 10, 5, c, gamma)
    computeProb(actualDS, DS1, DS2, DS3, c1, gamma1, c2, gamma2, c3, gamma3)

if __name__ == "__main__": main()