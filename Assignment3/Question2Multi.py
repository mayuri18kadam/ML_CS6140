import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import getcontext, Decimal
import math

getcontext().prec = 30


def getwordFreqList(data):
    X = np.zeros((len(np.unique(data[:, 0])), np.amax(np.unique(data[:, 1]))))
    for row in data:
        X[row[0] - 1, row[1] - 1] = row[2]
    wordFreq = np.sum(X, axis=0)
    vocabNew = np.vstack((np.unique(data[:, 1]), wordFreq)).T
    vocabNew = vocabNew[vocabNew[:, 1].argsort()[::-1]]
    return vocabNew


def getDocVSVocab(data, vocab):
    X = np.zeros((len(np.unique(data[:, 0])), np.amax(np.unique(data[:, 1]))))
    for row in data:
        if row[1] in vocab[:, 0]:
            X[row[0] - 1, row[1] - 1] = row[2]
    return X


def getDocVSClass_cntClass(label, map):
    y = np.zeros((len(label), len(np.unique(map[:, 1]))))
    i = 0
    for row in label:
        y[i, row - 1] = 1
        i = i + 1
    cntClass = np.sum(y, axis=0)
    return y, cntClass


def getTheta(X, y, cntClass, vocab):
    vocabLen = len(vocab)
    classLen = len(cntClass)
    theta = np.zeros((vocabLen, classLen))
    for j in range(vocabLen):
        for k in range(classLen):
            num = np.sum(np.multiply(X[:, j], y[:, k]))
            den = np.sum(np.multiply(np.sum(X, axis=1), y[:, k]))
            cal = Decimal((num + 1) / (den + 2))
            theta[j, k] = cal
    return theta


def getPredictedOutput(theta, cntClass, X):
    y_pred = []
    for n in range(len(X)):
        numArr = []
        for k in range(len(cntClass)):
            cal = 1
            for v in range(len(theta)):
                cal = Decimal(cal) * math.pow(Decimal(theta[v, k]), X[n, v])
            num = Decimal(cal) * Decimal(cntClass[k])
            numArr.append(num)

        y_pred.append(np.argmax(numArr) + 1)
    return y_pred


def getAccuracy(actual, pred):
    acc = 0
    for idx in range(len(actual)):
        if actual[idx] == pred[idx]:
            acc = acc + 1
    return (acc / len(actual)) * 100


def getPrecisionRecallByClass(actual, pred):
    preArr = []
    recArr = []
    classListPre, count1 = np.unique(actual, return_counts=True)
    classListRec, count2 = np.unique(pred, return_counts=True)
    for c in range(len(classListPre)):
        pre = 0
        for idx in range(len(actual)):
            if actual[idx] == classListPre[c] and actual[idx] == pred[idx]:
                pre = pre + 1
        preArr.append(pre / count1[c])

    for c in range(len(classListRec)):
        rec = 0
        for idx in range(len(actual)):
            if actual[idx] == classListRec[c] and actual[idx] == pred[idx]:
                rec = rec + 1
        recArr.append(rec / count2[c])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_title('Deliverable 2.5.1 : Precision VS Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Precision')
    ax.plot(classListPre, preArr, ls='--', marker='o', c='y', label='Precision VS Class')

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.set_title('Deliverable 2.5.1 : Recall VS Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Recall')
    ax.plot(classListRec, recArr, ls='--', marker='v', c='m', label='Recall VS Class')

    plt.show()
    return preArr, recArr


def evaluateAlgorithm(data, label, map, vocab, testData, testLabel, testMap):
    # wordFreq_size = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000]
    wordFreq_size = [10, 20, 30]

    accArr = []
    for l in wordFreq_size:
        vocabNew = getwordFreqList(data)
        X = getDocVSVocab(data, vocabNew[0:l, :])
        y, cntClass = getDocVSClass_cntClass(label, map)
        theta = getTheta(X, y, cntClass, vocabNew)

        X_test = getDocVSVocab(testData, vocabNew[0:l, :])
        y_pred_test = getPredictedOutput(theta, cntClass, X_test[700:710, :])
        acc = getAccuracy(testLabel[700:710, :], y_pred_test)
        accArr.append(acc)
        preArr, recArr = getPrecisionRecallByClass(testLabel[700:710, :], y_pred_test)

    print("Accuracy Arr = ", accArr)
    print("Precision Arr = ", preArr)
    print("Recal Arr = ", recArr)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_title('Deliverable 2.5.1 : Accuracy VS Vocab size')
    ax.set_xlabel('Vocab size')
    ax.set_ylabel('Accuracy')
    ax.plot(wordFreq_size, accArr, ls='--', marker='^', c='c', label='Accuracy VS Vocab size')
    plt.show()


def main():
    trainData = np.array(pd.read_csv('20news-bydate/matlab/train.data', sep=' ', header=None))
    trainLabel = np.array(pd.read_csv('20news-bydate/matlab/train.label', sep=' ', header=None))
    trainMap = np.array(pd.read_csv('20news-bydate/matlab/train.map', sep=' ', header=None))

    testData = np.array(pd.read_csv('20news-bydate/matlab/test.data', sep=' ', header=None))
    testLabel = np.array(pd.read_csv('20news-bydate/matlab/test.label', sep=' ', header=None))
    testMap = np.array(pd.read_csv('20news-bydate/matlab/test.map', sep=' ', header=None))

    vocab = np.array(pd.read_csv('vocabulary.txt', sep=' ', header=None))
    evaluateAlgorithm(trainData, trainLabel, trainMap, vocab, testData, testLabel, testMap)


if __name__ == "__main__": main()