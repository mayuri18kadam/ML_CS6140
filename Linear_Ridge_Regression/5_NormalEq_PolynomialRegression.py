'''
Created on Feb 9, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from math import sqrt
from numpy.linalg import inv
import matplotlib.pyplot as plt

def getWeights(X, y): 
    return np.matmul(np.matmul(inv(np.matmul(X.T, X)), X.T), y)
    
def getPredictedOutputAndWeights(X, y):
    # appending one's array
    # X_train_final is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols + 1), with 1st column of 1's
    X_final = np.append(np.ones((len(X), 1)), X, axis=1)        
    # bringing y_train_new in shape[[],[],...,[]] i.e. n*1
    y_final = []
    for ls in y:
        y_final.append([ls])            
    # initializing weights array w[[0], [0],...[0]] to size (orig_no_of_feature_cols + 1)*1
    w = np.zeros((len(X_final[0]), 1))        
    w = getWeights(X_final, y_final)
    y_pred = np.matmul(X_final, w)   
    return y_pred, w

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
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

def evaluate_algorithm(dataset, n_folds, p):
    rmse_train = []
    sse_train = []
    rmse_test = []
    sse_test = []
    kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(dataset):
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        # dataset_train_zscore is of shape[[],[],...[]] i.e. n*(orig_no_of_all_cols)
        meanArr = np.mean(dataset_train, axis=0)
        stdArr = np.std(dataset_train, axis=0)
        dataset_train_zscore = calZScore(dataset_train, meanArr, stdArr)
        dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)
              
        dataset_train_pow = addHigherPowerCols(dataset_train_zscore[:,0:-1], p)
        dataset_test_pow = addHigherPowerCols(dataset_test_zscore[:,0:-1], p)
        # X is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols)
        X_train, X_test = dataset_train_pow[:,0:-1], dataset_test_pow[:,0:-1]
        y_train, y_test = dataset_train[:,-1], dataset_test[:,-1]
                
        y_pred_train, w_train = getPredictedOutputAndWeights(X_train, y_train)
        sse_train_val = np.sum(np.square(y_pred_train - y_train[np.newaxis].T))
        rmse_train_val = sqrt(sse_train_val/len(y_train))
        rmse_train.append(rmse_train_val)
        sse_train.append(sse_train_val)
        
        X_test_final = np.append(np.ones((len(X_test), 1)), X_test, axis=1)
        y_pred_test = np.matmul(X_test_final, w_train)
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
        
def evaluateAlgorithmNoCrossValidation(dataset, datasetTest, p):
          
    # X is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols)
    X_train = addHigherPowerCols(dataset[:,0:-1], p)
    y_train = dataset[:,-1]
    X_test = addHigherPowerCols(datasetTest[:,0:-1], p)
    y_test = datasetTest[:,-1]
            
    y_pred_train, w_train = getPredictedOutputAndWeights(X_train, y_train)
    sse_train_val = np.sum(np.square(y_pred_train - y_train[np.newaxis].T))
    rmse_train_val = sqrt(sse_train_val/len(y_train))
    
    X_test_final = np.append(np.ones((len(X_test), 1)), X_test, axis=1)
    y_pred_test = np.matmul(X_test_final, w_train)
    sse_test_val = np.sum(np.square(y_pred_test - y_test[np.newaxis].T))
    rmse_test_val = sqrt(sse_test_val/len(y_test))
    
    print("Training RMSE = ",rmse_train_val)
    print("Training SSE = ",sse_train_val)
    print("Testing RMSE = ",rmse_test_val)
    print("Testing SSE = ",sse_test_val)
    return sse_train_val/len(X_train), sse_test_val/len(X_test_final)

def main():
    print("Sinusoid data: \n")
    sinusoid = pd.read_csv('sinData_Train.csv', sep=',', header=None)
    sinusoid.reindex(np.random.permutation(sinusoid.index))   
    sinusoid_test = pd.read_csv('sinData_Validation.csv', sep=',', header=None)
    sinusoid_test.reindex(np.random.permutation(sinusoid_test.index))   
    sinusoidTrainSSE = []
    sinusoidTestSSE = []
    sinusoid_P = np.arange(1, 16, 1)
    for p in sinusoid_P:
        print("for p = ",p)
        # evaluateAlgorithmNoCrossValidation(dataset_as_ndarray, valicationDataset_as_ndarray, power)
        sse_train_val, sse_test_val = evaluateAlgorithmNoCrossValidation(sinusoid.values, sinusoid_test.values, p)
        sinusoidTrainSSE.append(sse_train_val)
        sinusoidTestSSE.append(sse_test_val)
        
 
    print("\n\n")    
      
    print("Yacht data: \n")
    yacht = pd.read_csv('yachtData.csv', sep=',', header=None)
    yacht.reindex(np.random.permutation(yacht.index))   
    yachtTrainRMSE = []
    yachtTestRMSE = []
    yacht_P = np.arange(1, 8, 1)
    for p in yacht_P: 
        print("for p = ",p)
        # evaluate_algorithm(dataset_as_ndarray, n_folds, power)
        rmse_train_val, rmse_test_val = evaluate_algorithm(yacht.values, 10, p)  
        yachtTrainRMSE.append(rmse_train_val)
        yachtTestRMSE.append(rmse_test_val) 
        print("\n") 
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121)
    ax1.set_title('Sinusoid')
    ax1.set_xlabel('power of features')
    ax1.set_ylabel('SSE')
    ax1.plot(sinusoid_P, sinusoidTrainSSE, ls='--', marker='+', c='b', label='train sse')
    ax1.plot(sinusoid_P, sinusoidTestSSE, ls='--', marker='+', c='r', label='test rmse')
    
    ax2 = fig.add_subplot(122)
    ax2.set_title('Yacht')
    ax2.set_xlabel('power of features')
    ax2.set_ylabel('RMSE')
    ax2.plot(yacht_P, yachtTrainRMSE, ls='--', marker='+', c='b', label='train rmse')
    ax2.plot(yacht_P, yachtTestRMSE, ls='--', marker='+', c='r', label='test rmse')
    
    plt.legend(loc='upper left')
    plt.tight_layout(2, 2, 2)
    plt.show() 
    
    
if __name__ == "__main__": main()

