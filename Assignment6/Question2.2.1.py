
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

def evaluate_algorithm(dataset, T):
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.20, random_state=42)

    # dataset_train_zscore is of shape[[],[],...[]] i.e. n*(orig_no_of_all_cols)
    meanArr = np.mean(dataset_train, axis=0)
    stdArr = np.std(dataset_train, axis=0)
    dataset_train_zscore = calZScore(dataset_train, meanArr, stdArr)
    dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)

    feature, threshold = getBestFeatureMidpoint(dataset_train_zscore, (1/len(dataset_train_zscore)))

def main():
    print("breastcancer data:")
    breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    T = np.array([100, 200, 300, 400, 500])
    evaluate_algorithm(breastcancer.values, T)

if __name__ == "__main__": main()