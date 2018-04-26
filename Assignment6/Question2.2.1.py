
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

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

def calBestThreshold(feature, classCol, midpoints, instWgt) :
    feature1 = np.vstack((feature, classCol)).T
    feature1.sort(0)
    errorDiff = -1
    mid = -1
    for m in midpoints:
        error = 0
        rownum = 0
        for r in feature1:
            if (r[0]<m and r[1]<=0) or (r[0]>=m and r[1]>0):
                error = error+instWgt[rownum]
            rownum = rownum + 1
        if abs(0.5 - error) > errorDiff :
            errorDiff = abs(0.5 - error)
            mid = m
    return errorDiff, mid

def getBestFeatureMidpoint(xTrain, yTrain, instWgt):
    finalErrorDiff = -1
    finalMidpoint = -1
    finalFeature = -1
    for f in range(len(xTrain[0])):
        midpoints = calMidpoints(xTrain[:,f])
        errorDiff, midpoint = calBestThreshold(xTrain[:, f], yTrain, midpoints, instWgt)
        if errorDiff > finalErrorDiff:
            finalErrorDiff = errorDiff
            finalMidpoint = midpoint
            finalFeature = f
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
        featureIdx, threshold = getBestFeatureMidpoint(xTrain, yTrain, instWgt)
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
    # print("breastcancer data:")
    # breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    # T = [1, 2, 3, 4, 5]
    # for t in T:
    #     print("main t = ",t)
    #     testLocalErr, testErr, trainingErr = evaluate_algorithm(breastcancer.values, t)
    # t = 1: AccuracyScore = 89.47368421052632
    # t = 2: AccuracyScore = 89.47368421052632
    # t = 3: AccuracyScore = 100.0
    # t = 4: AccuracyScore = 100.0
    # t = 5: AccuracyScore = 100.0
    # Hence setting T=3 for breastcancer data
    # testLocalErr, testErr, trainingErr = evaluate_algorithm(breastcancer.values, 3)

    # print("diabetes data:")
    # diabetes = pd.read_csv('diabetes.csv', sep=',', header=None)
    # T = [1, 2, 3, 4, 5]
    # for t in T:
    #     print("main t = ",t)
    #     testLocalErr, testErr, trainingErr = evaluate_algorithm(diabetes.values, t)
    # t = 1: AccuracyScore = 88.31168831168831
    # t = 2: AccuracyScore = 88.31168831168831
    # t = 3: AccuracyScore = 100.0
    # t = 4: AccuracyScore = 100.0
    # t = 5: AccuracyScore = 100.0
    # Hence setting T=3 for diabetes data
    # testLocalErr, testErr, trainingErr = evaluate_algorithm(diabetes.values, 3)

    print("spambase data:")
    spambase = pd.read_csv('spambase.csv', sep=',', header=None)
    # T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # for t in T:
    #     print("main t = ",t)
    #     testLocalErr, testErr, trainingErr = evaluate_algorithm(spambase.values, t)
    # t = 1: AccuracyScore = 82.41042345276873
    # t = 2: AccuracyScore = 82.41042345276873
    # t = 3: AccuracyScore = 100.0
    # t = 4: AccuracyScore = 100.0
    # t = 5: AccuracyScore = 100.0
    # Hence setting T=3 for spambase data
    testLocalErr, testErr, trainingErr = evaluate_algorithm(spambase.values, 3)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_title('spambase data: testLocalErr')
    ax1.set_xlabel('T')
    ax1.set_ylabel('testLocalErr')
    ax1.plot(np.arange(1, len(testLocalErr)+1, 1), testLocalErr, ls='--', marker='o', c='m', label='testLocalErr')

    ax2 = fig.add_subplot(222)
    ax2.set_title('spambase data: testErr')
    ax2.set_xlabel('T')
    ax2.set_ylabel('testErr')
    ax2.plot(np.arange(1, len(testErr) + 1, 1), testErr, ls='--', marker='s', c='b', label='testErr')

    ax3 = fig.add_subplot(223)
    ax3.set_title('spambase data: trainingErr')
    ax3.set_xlabel('T')
    ax3.set_ylabel('trainingErr')
    ax3.plot(np.arange(1, len(trainingErr) + 1, 1), trainingErr, ls='--', marker='v', c='g', label='trainingErr')

    plt.legend(loc='upper left')
    plt.tight_layout(2, 2, 2)
    plt.show()

if __name__ == "__main__": main()