import numpy as np
import pandas as pd
import math
from random import randrange
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Node:
    def __init__(self, featureIndex, childNodes, midpoint, nodeName, parentCat, output):
        self.featureIndex = featureIndex
        self.childNodes = childNodes
        self.midpoint = midpoint
        self.nodeName = nodeName
        self.parentCat = parentCat
        self.output = output

# func: to compute midpoints for a numeric feature column
def calMidpoints(feature, classCol) :
    feature1 = np.vstack((feature,classCol)).T
    feature1.sort(0)
    midpoints = []
    prevClass = feature1[0,1]
    prevKey = feature1[0,0]
    count = 0
    for i in feature1[1:]:
        if i[1] != prevClass:
            midpoints.append((prevKey + i[0])/2)
            count = count + 1
            prevClass = i[1]
        prevKey = i[0]
    return midpoints

# func: calculates entrophy for output column 'classCol'
def calEntrophy(classCol) :
    uniqueClass, classCount = np.unique(classCol,return_counts=True)
    sum = np.sum(classCount)
    entrophy = 0
    for i in classCount:
        entrophy = (i/sum)*(math.log(i/sum,2)) + entrophy
    entrophy = -1 * entrophy
    return entrophy

# func: calculates best IG for a given feature and set of midpoints
def calIG(feature, classCol, midpoints) :
    featureEntrophy = calEntrophy(classCol)
    feature1 = np.vstack((feature, classCol)).T
    feature1.sort(0)
    total = len(feature1)
    IG = 0
    mid = 0
    for m in midpoints:
        lCount = 0
        rCount = 0
        lDSFeature = []
        lDSClassCol = []
        rDSFeature = []
        rDSClassCol = []
        for r in feature1:
            if r[0] < m :
                lCount = lCount + 1
                lDSFeature.append(r[0])
                lDSClassCol.append(r[1])
            else:
                rCount = rCount + 1
                rDSFeature.append(r[0])
                rDSClassCol.append(r[1])
        IGNew = featureEntrophy - ((lCount/total * calEntrophy(lDSClassCol)) + (rCount/total * calEntrophy(rDSClassCol)))
        if IG < IGNew :
            IG = IGNew
            mid = m
    return IG, m

# func: calculates best IG for a given categorical feature
def calIGCategorical(feature, classCol) :
    featureEntrophy = calEntrophy(classCol)
    feature1 = np.vstack((feature, classCol)).T
    feature1.sort(0)
    total = len(feature1)
    IG = 0
    entrophy = []

    featureCategory, count = np.unique(feature, return_counts=True)
    for cat in featureCategory:
        childClassCol = []
        for row in feature1:
            if row[0] == cat :
                childClassCol.append(row[1])
        entrophy.append(calEntrophy(childClassCol))
    sum = 0
    for i in range(len(featureCategory)):
        sum = sum + ((count[i]/total) * entrophy[i])
    IG = featureEntrophy - sum
    return IG, featureCategory

# func: determines the best feature and midpoint and returns the best root node and generates tree for it recursively
def getNode(colsCount, X_train, Y_train, nmin):
    if (len(X_train)<nmin or calEntrophy(Y_train)==0):
        outputCategory, count = np.unique(Y_train, return_counts=True)
        res = count.argmax()
        node = Node('', [], '', '', '', outputCategory[res])
        return node

    finalIG = 0
    finalMidpoint = 0
    finalFeature = 0
    Y_train_new = []
    IG = 0
    midpoint = 0
    finalFeatureCategory = []
    featureCategory = []
    for ls in Y_train:
        Y_train_new.append([ls])
    dataset = np.append(X_train, Y_train_new, axis=1)
    for f in range(colsCount-1):
        if np.issubdtype(type(X_train[0,f]), np.number):
            midpoints = calMidpoints(X_train[:, f], Y_train)
            IG, midpoint = calIG(X_train[:, f], Y_train, midpoints)
        else:
            IG, featureCategory = calIGCategorical(X_train[:,f], Y_train)
        if IG > finalIG :
            finalIG = IG
            finalMidpoint = midpoint
            finalFeature = f
            finalFeatureCategory = featureCategory

    allChilds = []
    if np.issubdtype(type(X_train[0, finalFeature]), np.number):
        lChild = []
        rChild = []
        for row in dataset:
            if row[finalFeature] < finalMidpoint:
                lChild.append(row)
            else :
                rChild.append(row)
        if len(lChild) != 0:
            allChilds.append(lChild)
        if len(rChild) != 0:
            allChilds.append(rChild)
    else:
        for cat in finalFeatureCategory:
            child = []
            for row in dataset:
                if row[finalFeature] == cat:
                    child.append(row)
            allChilds.append(child)
    nodeName = ''
    childNodes = []
    outputCategory, count = np.unique(Y_train, return_counts=True)
    res = count.argmax()
    if len(allChilds) == 0 or len(allChilds) == 1:
        return Node('', [], '', '', '', outputCategory[res])
    for child in allChilds:
        child = np.array(child)
        if(len(child)!=0):
            names, namesCount = np.unique(child[:, finalFeature], return_counts=True)
            maxCount = namesCount.argmax()
            nodeName = names[maxCount]
            newChild = getNode(colsCount, child[:,0:colsCount-1], child[:,colsCount-1], nmin)
            newChild.nodeName = nodeName
            childNodes.append(newChild)
        else:
            childNodes.append(Node('', [], '', '', '', outputCategory[res]))
    node = Node(finalFeature, childNodes, midpoint, nodeName, outputCategory[res], '')
    return node

# func: predicts result for a row
def predictVal(node, x) :
    res = ''
    if len(node.childNodes) == 0:
        return node.output
    for child in node.childNodes:
        if np.issubdtype(type(x[node.featureIndex]), np.number):
            if x[node.featureIndex] < node.midpoint:
                res = predictVal(child, x)
                if res != '':
                    return res
            else:
                res = predictVal(child, x)
                if res != '':
                    return res
        else:
            if x[node.featureIndex] == child.nodeName:
                res = predictVal(child,x)
                if res != '':
                    return res
    if res == '':
        return node.parentCat
    return res

# func: returns list of predicted values
def getPredictedOutput(node, features):
    Y_pred = []
    for f in features:
        #print("f = ",f)
        if f[0] == '0' or f[1] == '1':
            f = list(map(float,f))
        Y_pred.append(predictVal(node, f))
    return Y_pred

# func: computes accuracy score for predicted and actual data
def getAccuracyScore(test, pred):
    count = 0
    for i in range(len(test)):
        if test[i] == pred[i] :
            count = count + 1
    return count/len(test) * 100

# func: returns the best node and predicted and output results for a given training and test dataset
def getTree(dataset, testSet, nmin):
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

    accScore_test = getAccuracyScore(dataset_Y_test, Y_pred_test)
    accScore_train = getAccuracyScore(dataset_Y_train, Y_pred_train)

    return node, accScore_test, accScore_train, Y_pred_test, Y_pred_train

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
    bestAvgAccuracy = 0
    avg_test = []
    avg_train = []

    for min in nmin:
        print("For nmin = ",min)
        scores_test = list()
        scores_train = list()
        i = 0
        confMatrix = []
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
            finalNode, accuracy_test, accuracy_train, Y_pred_test, Y_pred_train = getTree(train_set_final, test_set_final, min)
            scores_test.append(accuracy_test)
            scores_train.append(accuracy_train)
            if len(confMatrix) == 0:
                confMatrix = confusion_matrix(test_set_final[:,-1], Y_pred_test)
            else :
                confMatrix = np.add(confMatrix,confusion_matrix(test_set_final[:,-1], Y_pred_test))
        print(confMatrix)

        standardDeviation_test = np.std(scores_test)
        standardDeviation_train = np.std(scores_train)
        avgAccuracy_test = np.average(scores_test)
        avgAccuracy_train = np.average(scores_train)
        avg_test.append(avgAccuracy_test)
        avg_train.append(avgAccuracy_train)
        if avgAccuracy_test > bestAvgAccuracy:
            bestAvgAccuracy = avgAccuracy_test
            bestNmin = min

        print("Accuracy test = ",scores_test)
        print("Standard Deviation test = ",standardDeviation_test)
        print("avg accuracy test = ",avgAccuracy_test)
        print("Accuracy train = ", scores_train)
        print("Standard Deviation train = ", standardDeviation_train)
        print("avg accuracy train = ", avgAccuracy_train)

    print("Best nmin is: ",bestNmin)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(nmin, avg_test, s=10, c='b', linestyle="dashed", marker="s", label='test accuracy')
    ax1.scatter(nmin, avg_train, s=10, c='r', linestyle="dashed", marker="o", label='train accuracy')
    plt.legend(loc='upper left')
    plt.show()


def main():
    mushroom_data = pd.read_csv('mushroom.csv', sep=',', header=None)
    mushroom_nmin = np.array([0.05, 0.1, 0.15])
    evaluate_algorithm(mushroom_data.values, 10, mushroom_nmin)
    print("Using binary one-hot encoding: ")
    mushroom_hotEncoded = np.array(pd.get_dummies(mushroom_data,columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
    X_mushroom_hotEncoded = mushroom_hotEncoded[:,1:len(mushroom_hotEncoded)-1]
    Y_mushroom_hotEncoded = mushroom_hotEncoded[:,0]
    Y_mushroom_hotEncoded_new = []
    for row in Y_mushroom_hotEncoded:
        Y_mushroom_hotEncoded_new.append([row])
    mushroom_hotEncoded_final = np.append(X_mushroom_hotEncoded,Y_mushroom_hotEncoded_new,axis=1)
    #evaluate_algorithm(mushroom_hotEncoded_final, 10, mushroom_nmin)
    evaluate_algorithm(mushroom_hotEncoded_final, 10, np.array([5, 10, 15]))

if __name__ == "__main__": main()