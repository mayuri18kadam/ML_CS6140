'''
Created on Feb 7, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from math import sqrt
import matplotlib.pyplot as plt

def getSSE(X, y, w):
    sse =  np.sum(np.square(np.matmul(X, w) - y))
    return sse

def getRMSE(X, y, w):
    rmse = sqrt(getSSE(X, y, w)/len(X))
    return rmse

def getWeights(X, y, w, max_iter, learning_rate, tolerance): 
    rmse = 0 
    for iterCount in range(max_iter):
        rmse_new = getRMSE(X, y, w)
        if abs(rmse_new - rmse) <= tolerance:
            return w
        for featureIndex in range(len(X[0])):
            s = 0
            for rowIndex in range(len(X)):
                s = s + ((np.matmul(X[rowIndex],w) - y[rowIndex]) * X[rowIndex, featureIndex])
            w[featureIndex] = w[featureIndex] - learning_rate*(s)
        rmse = rmse_new
    return w
    
def getPredictedOutputAndWeights(X, y, max_iter, learning_rate, tolerance):
    # appending one's array
    # X_train_final is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols + 1), with 1st column of 1's
    X_final = np.append(np.ones((len(X), 1)), X, axis=1)        
    # bringing y_train_new in shape[[],[],...,[]] i.e. n*1
    y_final = []
    for ls in y:
        y_final.append([ls])            
    # initializing weights array w[[0], [0],...[0]] to size (orig_no_of_feature_cols + 1)*1
    w = np.zeros((len(X_final[0]), 1))        
    w = getWeights(X_final, y_final, w, max_iter, learning_rate, tolerance)
    y_pred = np.matmul(X_final, w)   
    return y_pred, w

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
    return dataset

def evaluate_algorithm(dataset, n_folds, max_iter, learning_rate, tolerance):
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
        #print("dataset_train_zscore = ",dataset_train_zscore)
        dataset_test_zscore = calZScore(dataset_test, meanArr, stdArr)
              
        # X is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols)
        X_train, X_test = dataset_train_zscore[:,0:-1], dataset_test_zscore[:,0:-1]
        y_train, y_test = dataset_train[:,-1], dataset_test[:,-1]
                
        y_pred_train, w_train = getPredictedOutputAndWeights(X_train, y_train, max_iter, learning_rate, tolerance)
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

    return rmse_train, rmse_test
        

def main():
    print("Housing data: ")
    housing = pd.read_csv('housing.csv', sep=',', header=None)
    housing.reindex(np.random.permutation(housing.index))    
    # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
    rmse_train_housing, rmse_test_housing = evaluate_algorithm(housing.values, 10, 1000, 0.0004, 0.005)
    
    print("\n\n")    
     
    print("Yacht data: ")
    yacht = pd.read_csv('yachtData.csv', sep=',', header=None)
    yacht.reindex(np.random.permutation(yacht.index))    
    # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
    rmse_train_yacht, rmse_test_yacht = evaluate_algorithm(yacht.values, 10, 1000, 0.001, 0.001)
       
    print("\n\n")
       
    print("Concrete data: ")
    concrete = pd.read_csv('concreteData.csv', sep=',', header=None)
    concrete.reindex(np.random.permutation(concrete.index))    
    # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
    rmse_train_concrete, rmse_test_concrete = evaluate_algorithm(concrete.values, 10, 1000, 0.0007, 0.0001) 
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(221)
    ax1.set_title('Housing')
    ax1.set_xlabel('folds')
    ax1.set_ylabel('RMSE')
    ax1.plot(np.arange(1, 11, 1), rmse_train_housing, ls='--', marker='+', c='b', label='train rmse')
    ax1.plot(np.arange(1, 11, 1), rmse_test_housing, ls='--', marker='+', c='r', label='test rmse')
    
    ax2 = fig.add_subplot(222)
    ax2.set_title('Yacht')
    ax2.set_xlabel('folds')
    ax2.set_ylabel('RMSE')
    ax2.plot(np.arange(1, 11, 1), rmse_train_yacht, ls='--', marker='+', c='b', label='train rmse')
    ax2.plot(np.arange(1, 11, 1), rmse_test_yacht, ls='--', marker='+', c='r', label='test rmse')
    
    ax3 = fig.add_subplot(223)
    ax3.set_title('Concrete')
    ax3.set_xlabel('folds')
    ax3.set_ylabel('RMSE')
    ax3.plot(np.arange(1, 11, 1), rmse_train_concrete, ls='--', marker='+', c='b', label='train rmse')
    ax3.plot(np.arange(1, 11, 1), rmse_test_concrete, ls='--', marker='+', c='r', label='test rmse')
    
    plt.legend(loc='upper left')
    plt.tight_layout(2, 2, 2)
    plt.show()   
    
if __name__ == "__main__": main()

