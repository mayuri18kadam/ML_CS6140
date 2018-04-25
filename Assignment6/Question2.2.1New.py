
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
import math

# func: to compute midpoints for a feature column
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

# func: calculates best IG for a given feature and set of midpoints
def calBestThreshold(feature, classCol, midpoints, instWgt) :
    # print("midpoints = ",len(midpoints))
    feature1 = np.vstack((feature, classCol)).T
    feature1.sort(0)
    errorDiff = -1
    mid = -1
    for m in midpoints:
        error = 0
        for r in feature1:
            if (r[0]<m and r[1]==-1) or (r[0]>=m and r[1]==1):
                error = error+instWgt
        if abs(0.5 - error) > errorDiff :
            errorDiff = abs(0.5 - error)
            mid = m
    # print("errorDiff = ", errorDiff)
    # print("mid = ", mid)
    return errorDiff, mid

def getBestFeatureMidpoint(dataset_train_zscore, instWgt):
    finalErrorDiff = -1
    finalMidpoint = -1
    finalFeature = -1
    for f in range(len(dataset_train_zscore[0])-1):
        midpoints = calMidpoints(dataset_train_zscore[:,f])
        errorDiff, midpoint = calBestThreshold(dataset_train_zscore[:, f], dataset_train_zscore[:,-1], midpoints, instWgt)
        if errorDiff > finalErrorDiff:
            finalErrorDiff = errorDiff
            finalMidpoint = midpoint
            finalFeature = f
    # print("finalFeature = ",finalFeature)
    # print("finalMidpoint = ", finalMidpoint)
    return finalFeature, finalMidpoint

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
    return dataset

def getError(dataset_train_zscore, featureIdx, threshold, instWgt):
    y_pred = []
    for row in dataset_train_zscore:
        if row[featureIdx]<threshold :
            y_pred.append(1)
        else:
            y_pred.append(-1)
    error = 0
    for n in range(len(y_pred)):
        if y_pred[n] != dataset_train_zscore[n,-1]:
            error = error + instWgt
    return error, y_pred

def getNormalizationConst(y_actual, y_pred, instWgt, beta):
    sum = 0
    loss = 0
    for n in range(len(y_actual)):
        expForZ = math.exp(-1 * beta * y_actual[n] * y_pred[n])
        expForLoss = math.exp(-1 * y_actual[n] * y_pred[n])
        loss = loss + expForLoss*expForZ
        sum = sum + instWgt*expForZ
    return sum, loss

def calEdge(dataset_train_zscore, featureIdx, threshold, instWgt):
    edge = 0
    for n in range(len(dataset_train_zscore)):
        actualOutput = 1 if dataset_train_zscore[n, featureIdx] < threshold else -1
        edge = edge + instWgt*(dataset_train_zscore[n,-1]*actualOutput)
    return edge

def evaluate_algorithm(dataset, T, tol):
    # print("T = ",T)
    # print("range of T = ", np.arange(1, T+1, 1))
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.20, random_state=42)
    meanArr = np.mean(dataset_train, axis=0)
    stdArr = np.std(dataset_train, axis=0)
    dataset_train_zscore = calZScore(dataset_train, meanArr, stdArr)
    dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)
    xTrain, yTrain = dataset_train_zscore[:,0:-1], dataset_train_zscore[:,-1]
    xTest, yTest = dataset_test_zscore[:, 0:-1], dataset_test_zscore[:, -1]

    instWgt = (1/len(dataset_train_zscore))
    alphaArr = []
    featureIdxArr = []
    thresholdArr = []
    lossOld = 0
    for t in np.arange(1, T+1, 1):
        print("t = ",t)
        print("instWgt = ",instWgt)
        featureIdx, threshold = getBestFeatureMidpoint(dataset_train_zscore, instWgt)
        featureIdxArr.append(featureIdx)
        thresholdArr.append(threshold)
        edge = calEdge(dataset_train_zscore, featureIdx, threshold, instWgt)
        print("edge = ",edge)
        alpha = math.log1p(abs((1+edge)/(1-edge)))
        alphaArr.append(alpha)
        error, y_pred = getError(dataset_train_zscore, featureIdx, threshold, instWgt)
        z, lossNew = getNormalizationConst(dataset_train_zscore[:, -1], y_pred, instWgt, alpha)
        instWgt = (instWgt * math.exp(-1 * alpha * np.sum(np.dot(y_pred, dataset_train_zscore[:,-1]))))/z
        print("lossNew = ",lossNew)
        print("lossOld = ",lossOld)
        print("abs(lossNew - lossOld) = ",abs(lossNew - lossOld))
        if abs(lossNew - lossOld) <= tol:
            break;
        lossOld = lossNew

    print("alphaArr = ",alphaArr)
    print("featureIdxArr = ", featureIdxArr)
    print("thresholdArr = ", thresholdArr)

def main():
    print("breastcancer data:")
    breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    evaluate_algorithm(breastcancer.values, 25, 0.0001)

if __name__ == "__main__": main()