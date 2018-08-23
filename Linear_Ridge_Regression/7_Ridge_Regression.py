'''
Created on Feb 11, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from math import sqrt
from numpy.linalg import inv
import matplotlib.pyplot as plt

def getWeights(X, y, l): 
    I = np.identity(len(X[0]))
    return np.matmul(np.matmul(inv(np.matmul(X.T, X) + l*I), X.T), y)
    
def getPredictedOutputAndWeights(X, y, l):       
    # bringing y_train_new in shape[[],[],...,[]] i.e. n*1
    y_final = []
    for ls in y:
        y_final.append([ls])  
    w_constant = np.mean(y)          
    # initializing weights array w[[0], [0],...[0]] to size (orig_no_of_feature_cols)*1
    w = np.zeros((len(X[0]), 1))        
    w = getWeights(X, y_final, l)
    y_pred = np.matmul(X, w) + w_constant
    return y_pred, w, w_constant

def centeringData(dataset, meanArr):
    for i in range(len(dataset[0])):
        for j in range(len(dataset)):
            dataset[j, i] = dataset[j, i] - meanArr[i]
    return dataset

def addHigherPowerCols(dataset, p):
    newDataset = dataset
    if p < 1:
        for i in range(p, 1):
            powDataset = np.power(dataset, i)
            newDataset = np.append(newDataset, powDataset, axis=1)
        return newDataset
    else:
        for i in range(1, p):
            powDataset = np.power(dataset, i+1)
            newDataset = np.append(newDataset, powDataset, axis=1)
        return newDataset

def evaluate_algorithm(dataset, n_folds, l, p):
    rmse_train = []
    sse_train = []
    rmse_test = []
    sse_test = []
    kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(dataset):
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        # dataset_train_zscore is of shape[[],[],...[]] i.e. n*(orig_no_of_all_cols)
        meanArr = np.mean(dataset_train, axis=0)
        dataset_train_centered = centeringData(dataset_train, meanArr)
        dataset_test_centered = centeringData(dataset_test, meanArr)
              
        X_dataset_train_pow = addHigherPowerCols(dataset_train_centered[:,0:-1], p)
        X_dataset_test_pow = addHigherPowerCols(dataset_test_centered[:,0:-1], p)
        # X is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols)
        X_train, X_test = X_dataset_train_pow, X_dataset_test_pow
        y_train, y_test = dataset_train_centered[:,-1], dataset_test_centered[:,-1]
                
        y_pred_train, w_train, w_constant = getPredictedOutputAndWeights(X_train, y_train, l)
        sse_train_val = np.sum(np.square(y_pred_train - y_train[np.newaxis].T))
        rmse_train_val = sqrt(sse_train_val/len(y_train))
        rmse_train.append(rmse_train_val)
        sse_train.append(sse_train_val)
        
        y_pred_test = np.matmul(X_test, w_train) + w_constant
        sse_test_val = np.sum(np.square(y_pred_test - y_test[np.newaxis].T))
        rmse_test_val = sqrt(sse_test_val/len(y_test))
        rmse_test.append(rmse_test_val)
        sse_test.append(sse_test_val)
    
    print("Training RMSE = ",rmse_train)
    print("avg Training RMSE = ",np.average(rmse_train))
    print("Training SSE = ",sse_train)
    print("avg training SSE = ",np.average(sse_train))
    print("Training RMSE Standard deviation = ",np.std(rmse_train))
    print("Training SSE Standard deviation = ",np.std(sse_train))
    print("")
    print("Testing RMSE = ",rmse_test)
    print("avg Testing RMSE = ",np.average(rmse_test))
    print("Testing SSE = ",sse_test)
    print("avg testing SSE = ",np.average(sse_test))
    print("Testing RMSE Standard deviation = ",np.std(rmse_test))
    print("Testing SSE Standard deviation = ",np.std(sse_test))
    return np.average(rmse_train), np.average(rmse_test)
        

def main():
    print("Sinusoid data: \n")
    sinusoid = pd.read_csv('sinData_Train.csv', sep=',', header=None)
    sinusoid.reindex(np.random.permutation(sinusoid.index))    
    lambdaArr = np.arange(0, 10.2, 0.2)
    
    print("for p = 5")
    rmse_train5 = []
    rmse_test5 = []
    for l in lambdaArr:
        print("for lambda = ",l)
        rmse_train1, rmse_test1 = evaluate_algorithm(sinusoid.values, 10, l, 5)
        rmse_train5.append(rmse_train1)
        rmse_test5.append(rmse_test1)
        
    print("\n")
    
    print("for p = 9")
    print()
    rmse_train9 = []
    rmse_test9 = []
    for l in lambdaArr:
        print("for lambda = ",l)
        rmse_train2, rmse_test2 = evaluate_algorithm(sinusoid.values, 10, l, 9)
        rmse_train9.append(rmse_train2)
        rmse_test9.append(rmse_test2)
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121)
    ax1.set_title('Sinusoid for p=5')
    ax1.set_xlabel('lambda')
    ax1.set_ylabel('RMSE')
    ax1.plot(lambdaArr, rmse_train5, ls='--', marker='+', c='b', label='train rmse')
    ax1.plot(lambdaArr, rmse_test5, ls='--', marker='+', c='r', label='test rmse')
    
    ax2 = fig.add_subplot(122)
    ax2.set_title('Sinusoid for p=9')
    ax2.set_xlabel('lambda')
    ax2.set_ylabel('RMSE')
    ax2.plot(lambdaArr, rmse_train9, ls='--', marker='+', c='b', label='train rmse')
    ax2.plot(lambdaArr, rmse_test9, ls='--', marker='+', c='r', label='test rmse')
    
    plt.legend(loc='upper left')
    plt.tight_layout(2, 2, 2)
    plt.show() 
    
    
if __name__ == "__main__": main()

