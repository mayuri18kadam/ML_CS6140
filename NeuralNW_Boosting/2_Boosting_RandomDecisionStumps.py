
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import random

def calMidpoints(feature) :
    featureUnique = np.unique(feature)
    featureUnique.sort()
    midpoints = np.zeros(len(feature))
    prevVal = featureUnique[0]
    count = 0
    for i in featureUnique:
        midpoints[count] = (prevVal+i)/2
        count = count + 1
        prevVal = i
    midpointsFinal = np.unique(midpoints)
    return midpointsFinal

def getBestFeatureMidpoint(xTrain):
    finalFeature = random.choice(range(len(xTrain[0])))
    finalMidpoint = random.choice(calMidpoints(xTrain[:, finalFeature]))
    return finalFeature, finalMidpoint

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
    return dataset

def getError(y_pred, y_actual, instWgt):
    error = 0
    for n in range(len(y_pred)):
        if y_pred[n] != y_actual[n]:
            error = error + instWgt[n]
    return error

def calAccuracyScore(y_pred, y_actual):
    return sum(y_pred != y_actual)*100 / float(len(y_actual))

def calPredictedOutput(X, featureIdx, threshold):
    y_pred = []
    for row in X:
        if row[featureIdx]<threshold :
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred

def getNormalizationConst(y_actual, y_pred, instWgt, beta):
    sum = 0
    for n in range(len(y_actual)):
        expForZ = math.exp(-1 * beta * y_actual[n] * y_pred[n])
        sum = sum + instWgt[n]*expForZ
    return sum

def calEdge(xTrain, yTrain, featureIdx, threshold, instWgt):
    edge = 0
    for n in range(len(xTrain)):
        actualOutput = 1 if xTrain[n, featureIdx] < threshold else -1
        edge = edge + instWgt[n]*(yTrain[n]*actualOutput)
    return edge

def getFinalPrediction(xTest, alphaArr, featureIdxArr, thresholdArr):
    y_predTrain = np.zeros(len(xTest))
    for n in range(len(alphaArr)):
        temp = [float(alphaArr[n])*int(x) for x in calPredictedOutput(xTest, featureIdxArr[n], thresholdArr[n])]
        y_predTrain = y_predTrain + temp
    return np.sign(y_predTrain)

def evaluate_algorithm(dataset, T):
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.20, random_state=42)
    meanArr = np.mean(dataset_train, axis=0)
    stdArr = np.std(dataset_train, axis=0)
    dataset_train_zscore = calZScore(dataset_train, meanArr, stdArr)
    dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)
    xTrain, yTrain = dataset_train_zscore[:,0:-1], dataset_train_zscore[:,-1]
    xTest, yTest = dataset_test_zscore[:, 0:-1], dataset_test_zscore[:, -1]

    # Initialize
    instWgt = np.ones(len(xTrain)) / len(xTrain)
    testLocalErr = []
    testErr = []
    trainingErr = []
    alphaArr = []
    featureIdxArr = []
    thresholdArr = []

    for t in np.arange(1, T+1, 1):
        print("t = ",t)
        # fit a simple decision tree, compute error on that
        featureIdx, threshold = getBestFeatureMidpoint(xTrain)
        y_predTrain = calPredictedOutput(xTrain, featureIdx, threshold)
        y_predTest = calPredictedOutput(xTest, featureIdx, threshold)
        testLocalErr.append(getError(y_predTest, yTest, instWgt))
        featureIdxArr.append(featureIdx)
        thresholdArr.append(threshold)

        edge = calEdge(xTrain, yTrain, featureIdx, threshold, instWgt)
        alpha = math.log1p(abs((1+edge)/(1-edge)))
        alphaArr.append(alpha)
        z = getNormalizationConst(yTrain, y_predTrain, instWgt, alpha)
        instWgt = np.multiply(instWgt, np.exp(-1 * alpha * np.exp(-1 * alpha * np.prod((y_predTrain, yTrain), axis=0))))/z
        testErr.append(getError(getFinalPrediction(xTest, alphaArr, featureIdxArr, thresholdArr), yTest, instWgt))
        trainingErr.append(getError(getFinalPrediction(xTrain, alphaArr, featureIdxArr, thresholdArr), yTrain, instWgt))

    y_test_final = getFinalPrediction(xTest, alphaArr, featureIdxArr, thresholdArr)
    print("Accuracy Score = ",calAccuracyScore(y_test_final, yTest))
    return testLocalErr, testErr, trainingErr

def main():
    print("breastcancer data:")
    breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    # T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # for t in T:
    #     print("main t = ",t)
    #     testLocalErr, testErr, trainingErr = evaluate_algorithm(breastcancer.values, t)
    # t = 1: AccuracyScore = 100.0
    # t = 2: AccuracyScore = 72.80701754385964
    # t = 3: AccuracyScore = 99.12280701754386
    # t = 4: AccuracyScore = 97.36842105263158
    # t = 5: AccuracyScore = 100.0
    # t = 6: AccuracyScore = 100.0
    # t = 7: AccuracyScore = 83.33333333333333
    # t = 8: AccuracyScore = 95.6140350877193
    # t = 9: AccuracyScore = 91.2280701754386
    # t = 10: AccuracyScore = 95.6140350877193
    # t = 11: AccuracyScore = 95.6140350877193
    # t = 12: AccuracyScore = 98.24561403508773
    # t = 13: AccuracyScore = 79.82456140350877
    # t = 14: AccuracyScore = 100.0
    # t = 15: AccuracyScore = 98.24561403508773
    # t = 16: AccuracyScore = 91.2280701754386
    # t = 17: AccuracyScore = 91.2280701754386
    # t = 18: AccuracyScore = 100.0
    # t = 19: AccuracyScore = 87.71929824561404
    # t = 20: AccuracyScore = 100.0
    # Hence setting T=14 for breastcancer data
    testLocalErr, testErr, trainingErr = evaluate_algorithm(breastcancer.values, 14)

    # print("diabetes data:")
    # diabetes = pd.read_csv('diabetes.csv', sep=',', header=None)
    # T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # for t in T:
    #     print("main t = ",t)
    #     testLocalErr, testErr, trainingErr = evaluate_algorithm(diabetes.values, t)
    # t = 1: AccuracyScore = 82.46753246753246
    # t = 2: AccuracyScore = 70.77922077922078
    # t = 3: AccuracyScore = 92.85714285714286
    # t = 4: AccuracyScore = 64.28571428571429
    # t = 5: AccuracyScore = 84.41558441558442
    # t = 6: AccuracyScore = 77.92207792207792
    # t = 7: AccuracyScore = 81.16883116883118
    # t = 8: AccuracyScore = 76.62337662337663
    # t = 9: AccuracyScore = 70.12987012987013
    # t = 10: AccuracyScore = 66.23376623376623
    # t = 11: AccuracyScore = 70.77922077922078
    # t = 12: AccuracyScore = 68.83116883116882
    # t = 13: AccuracyScore = 85.71428571428571
    # t = 14: AccuracyScore = 69.48051948051948
    # t = 15: AccuracyScore = 66.23376623376623
    # t = 16: AccuracyScore = 68.83116883116882
    # t = 17: AccuracyScore = 65.58441558441558
    # t = 18: AccuracyScore = 64.28571428571429
    # t = 19: AccuracyScore = 66.88311688311688
    # t = 20: AccuracyScore = 79.22077922077922
    # Hence setting T=3 for diabetes data
    # testLocalErr, testErr, trainingErr = evaluate_algorithm(diabetes.values, 3)

    # print("spambase data:")
    # spambase = pd.read_csv('spambase.csv', sep=',', header=None)
    # T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # for t in T:
    #     print("main t = ",t)
    #     testLocalErr, testErr, trainingErr = evaluate_algorithm(spambase.values, t)
    # t = 1: AccuracyScore = 57.871878393051034
    # t = 2: AccuracyScore = 58.19761129207383
    # t = 3: AccuracyScore = 57.76330076004343
    # t = 4: AccuracyScore = 57.65472312703583
    # t = 5: AccuracyScore = 61.99782844733985
    # t = 6: AccuracyScore = 58.957654723127035
    # t = 7: AccuracyScore = 57.65472312703583
    # t = 8: AccuracyScore = 60.26058631921824
    # t = 9: AccuracyScore = 57.65472312703583
    # t = 10: AccuracyScore = 57.65472312703583
    # t = 11: AccuracyScore = 57.65472312703583
    # t = 12: AccuracyScore = 57.65472312703583
    # t = 13: AccuracyScore = 57.65472312703583
    # t = 14: AccuracyScore = 57.65472312703583
    # t = 15: AccuracyScore = 57.65472312703583
    # t = 16: AccuracyScore = 57.65472312703583
    # t = 17: AccuracyScore = 57.65472312703583
    # t = 18: AccuracyScore = 57.65472312703583
    # t = 19: AccuracyScore = 57.65472312703583
    # t = 20: AccuracyScore = 57.65472312703583
    # Hence setting T=5 for spambase data
    # testLocalErr, testErr, trainingErr = evaluate_algorithm(spambase.values, 5)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_title('testLocalErr')
    ax1.set_xlabel('T')
    ax1.set_ylabel('testLocalErr')
    ax1.plot(np.arange(1, len(testLocalErr)+1, 1), testLocalErr, ls='--', marker='o', c='m', label='testLocalErr')

    ax2 = fig.add_subplot(222)
    ax2.set_title('testErr')
    ax2.set_xlabel('T')
    ax2.set_ylabel('testErr')
    ax2.plot(np.arange(1, len(testErr) + 1, 1), testErr, ls='--', marker='s', c='b', label='testErr')

    ax3 = fig.add_subplot(223)
    ax3.set_title('trainingErr')
    ax3.set_xlabel('T')
    ax3.set_ylabel('trainingErr')
    ax3.plot(np.arange(1, len(trainingErr) + 1, 1), trainingErr, ls='--', marker='v', c='g', label='trainingErr')

    plt.legend(loc='upper left')
    plt.tight_layout(2, 2, 2)
    plt.show()

if __name__ == "__main__": main()