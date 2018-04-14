import numpy as np
import pandas as pd
import math
from random import randrange
import matplotlib.pyplot as plt

# def: class Node
class Node:
    def __init__(self, featureIndex, lChild, rChild, midpoint, output):
        self.featureIndex = featureIndex
        self.lChild = lChild
        self.rChild = rChild
        self.midpoint = midpoint
        self.output = output

# func: calculates sum of squared error for output column 'classCol'
def calSSE(classCol):
    Gm = 0
    for y in classCol:
        Gm = Gm + y
    Gm = Gm/len(classCol)
    Em = 0
    for y in classCol:
        Em = Em + math.pow((y - Gm), 2)
    return Em

# func: computes Em, Gm for a given feature and all possible midpoints and returns the best Em and midpoint for that feature
def calLeastSSE(feature, classCol, midpoints) :
    if len(classCol) != 0:
        featureEm = calSSE(classCol)
    else:
        return 0, 0
    feature1 = np.vstack((feature, classCol)).T
    feature1.sort(0)
    total = len(feature1)
    parentEmFinal = 0
    mid = 0
    for m in midpoints:
        lCount = 0
        rCount = 0
        lDSClassCol = []
        rDSClassCol = []
        lEm = 0
        rEm = 0
        parentEm = 0
        for r in feature1:
            if r[0] <= m :
                lCount = lCount + 1
                lDSClassCol.append(r[1])
            else:
                rCount = rCount + 1
                rDSClassCol.append(r[1])
        if len(lDSClassCol) != 0:
            lEm = calSSE(lDSClassCol)
        else:
            lEm = 0
        if len(rDSClassCol) != 0:
            rEm = calSSE(rDSClassCol)
        else:
            rEm = 0
        childEm = (lCount/total * lEm) + (rCount/total * rEm)
        parentEm = featureEm - childEm
        if parentEm > parentEmFinal:
            parentEmFinal = parentEm
            mid = m
    return parentEmFinal, m

# func: determines the best feature and midpoint and returns the best root node and generates tree for it recursively
def getNode(colsCount, X_train, Y_train, nmin):
    Em = calSSE(Y_train)
    if len(X_train)<nmin:
        node = Node('', '', '', '', np.average(Y_train))
        return node

    finalEm = 0
    finalMidpoint = 0
    finalFeature = 0
    Y_train_new = []
    for ls in Y_train:
        Y_train_new.append([ls])
    dataset = np.append(X_train, Y_train_new, axis=1)

    for f in range(colsCount-1):
        midpoints = X_train[:, f]
        IG, midpoint = calLeastSSE(X_train[:, f], Y_train, midpoints)
        if IG > finalEm :
            finalEm = IG
            finalMidpoint = midpoint
            finalFeature = f

    lChild = []
    rChild = []
    for row in dataset:
        if row[finalFeature] < finalMidpoint:
            lChild.append(row)
        else :
            rChild.append(row)

    lChild = np.array(lChild)
    rChild = np.array(rChild)
    if len(lChild != 0) :
        lChild = getNode(colsCount, lChild[:,0:colsCount-1], lChild[:,colsCount-1], nmin)
    else :
        outputCategory, count = np.unique(Y_train, return_counts=True)
        res = count.argmax()
        return Node('', '', '', '', outputCategory[res])
    if len(rChild != 0) :
        rChild = getNode(colsCount, rChild[:, 0:colsCount - 1], rChild[:, colsCount - 1], nmin)
    else :
        outputCategory, count = np.unique(Y_train, return_counts=True)
        res = count.argmax()
        return Node('', '', '', '', outputCategory[res])
    node = Node(finalFeature, lChild, rChild, midpoint, '')
    return node

# func: predicts result for a row
def predictVal(node, x) :
    res = ''
    if node.rChild == '' and node.lChild == '':
        return node.output
    if x[node.featureIndex] < node.midpoint:
        res = predictVal(node.lChild, x)
        if res != '':
            return res
    else:
        res = predictVal(node.rChild, x)
        if res != '':
            return res
    return res

# func: returns list of predicted values
def getPredictedOutput(node, features):
    Y_pred = []
    for f in features:
        x = list(map(float,f))
        Y_pred.append(predictVal(node, x))
    return Y_pred

# func: computes mean squared error for predicted and actual data
def getMeanSquaredError(test, pred):
    sum = 0
    for i in range(len(test)):
        sum = sum + math.pow((test[i] - pred[i]), 2)
    #return math.sqrt(sum)
    return sum/len(test)

# func: returns the best node and predicted and output results for a given training and test dataset
def getBestTree(dataset, testSet, nmin):
    dataset_rowsCount = len(dataset)
    dataset_colsCount = len(dataset[0])
    dataset_X_train = dataset[:, 0:dataset_colsCount - 1]
    dataset_Y_train = dataset[:, dataset_colsCount - 1]
    dataset_X_test = testSet[:, 0:dataset_colsCount - 1]
    dataset_Y_test = testSet[:, dataset_colsCount - 1]
    dataset_nmin = np.round(nmin * dataset_rowsCount)

    node = getNode(dataset_colsCount, dataset_X_train, dataset_Y_train, dataset_nmin)
    Y_pred_test = np.array(getPredictedOutput(node, dataset_X_test))
    Y_pred_train = np.array(getPredictedOutput(node, dataset_X_train))

    mse_test = getMeanSquaredError(dataset_Y_test, Y_pred_test)
    mse_train = getMeanSquaredError(dataset_Y_train, Y_pred_train)

    return node, mse_test, mse_train

# func: for a complete dataset and count of folds, returns a list of datasets split into the number of folds
def kFoldCrossValidation(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# func: the main algorithm that takes a dataset as input and generates tree, and computes accuracy for each fold in k-fold cross validation
def evaluate_algorithm(dataset, n_folds, nmin):
    folds = kFoldCrossValidation(dataset, n_folds)
    bestNmin = 0
    bestMSE = 100
    avg_test = []
    avg_train = []

    for min in nmin:
        print("For nmin = ",min)
        scores_test = list()
        scores_train = list()
        i = 0
        while i < n_folds:
            train_set = list(folds)
            train_set.pop(i)
            train_set = sum(train_set, [])
            test_set = list()
            for row in folds[i]:
                row_copy = list(row)
                test_set.append(row_copy)
            i = i + 1
            test_set_final = np.array(test_set)
            train_set_final = np.array(train_set)
            finalNode, mse_test, mse_train = getBestTree(train_set_final, test_set_final, min)
            scores_test.append(mse_test)
            scores_train.append(mse_train)

        standardDeviation_test = np.std(scores_test)
        standardDeviation_train = np.std(scores_train)
        avgMSE_test = np.average(scores_test)
        avgMSE_train = np.average(scores_train)
        avg_test.append(avgMSE_test)
        avg_train.append(avgMSE_train)

        if avgMSE_test < bestMSE:
            bestMSE = avgMSE_test
            bestNmin = min

        print("MSE test = ",scores_test)
        print("Standard Deviation test = ",standardDeviation_test)
        print("avg MSE test = ",avgMSE_test)

        print("MSE train = ", scores_train)
        print("Standard Deviation train = ", standardDeviation_train)
        print("avg MSE train = ", avgMSE_train)

    print("Best nmin is: ",bestNmin)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(nmin, avg_test, s=10, c='b', linestyle="dashed", marker="s", label='test accuracy')
    ax1.scatter(nmin, avg_train, s=10, c='r', linestyle="dashed", marker="o", label='train accuracy')
    plt.legend(loc='upper left')
    plt.show()

def main():
    print("Housing data:")
    housing_data = pd.read_csv('housing_normalized.csv', sep=',', header=None)
    housing_nmin = np.array([0.05, 0.1, 0.15, 0.2])
    #housing_nmin = np.array([0.05])
    evaluate_algorithm(housing_data.values, 10, housing_nmin)

if __name__ == "__main__": main()