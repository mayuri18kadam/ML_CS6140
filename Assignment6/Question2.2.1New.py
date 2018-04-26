
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
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
    return sum(y_pred != y_actual) / float(len(y_actual))

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
    trainErr = []
    testErr = []
    alphaArr = []
    featureIdxArr = []
    thresholdArr = []

    for t in np.arange(1, T+1, 1):
        print("t = ",t)
        # fit a simple decision tree, compute error on that
        featureIdx, threshold = getBestFeatureMidpoint(xTrain, yTrain, instWgt)
        y_predTrain = calPredictedOutput(xTrain, featureIdx, threshold)
        y_predTest = calPredictedOutput(xTest, featureIdx, threshold)
        trainErr1 = getError(y_predTrain, yTrain, instWgt)
        testErr1 = getError(y_predTest, yTest, instWgt)
        trainErr.append(trainErr1)
        testErr.append(testErr1)
        featureIdxArr.append(featureIdx)
        thresholdArr.append(threshold)

        edge = calEdge(xTrain, yTrain, featureIdx, threshold, instWgt)
        alpha = math.log1p(abs((1+edge)/(1-edge)))
        alphaArr.append(alpha)
        z = getNormalizationConst(yTrain, y_predTrain, instWgt, alpha)
        instWgt = np.multiply(instWgt, np.exp(-1 * alpha * np.exp(-1 * alpha * np.prod((y_predTrain, yTrain), axis=0))))/z

    y_test_final = getFinalPrediction(xTest, alphaArr, featureIdxArr, thresholdArr)
    print("y_test_final = ",y_test_final)
    print("alphaArr = ",alphaArr)
    print("featureIdxArr = ", featureIdxArr)
    print("thresholdArr = ", thresholdArr)
    print("trainErr = ", trainErr)
    print("testErr = ", testErr)
    print("Accuracy Score = ",calAccuracyScore(y_test_final, yTest))

def main():
    print("breastcancer data:")
    breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    evaluate_algorithm(breastcancer.values, 25)

if __name__ == "__main__": main()