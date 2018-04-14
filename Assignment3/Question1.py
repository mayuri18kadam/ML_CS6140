'''
Created on Feb 8, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from math import log1p
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
# import matplotlib.pyplot as plt
    
def calSigmoid(val):
    sigmoid = (1/(1 + np.exp(val*-1)))
    return sigmoid
    
def calLossFunc(X, y, w):
    loss = 0    
    sigArr = []
    for rowIndex in range(len(X)):
        sig = calSigmoid(np.matmul(X[rowIndex],w))
        sigArr.append(sig)
        loss = loss - (y[rowIndex] * np.array(log1p(sig)) + (np.array(1) - y[rowIndex])*np.array(log1p(1 - sig)))
    return loss, sigArr
    
def getWeightsGradDecent(X, y, w, max_iter, learning_rate, tolerance): 
    loss = 0 
    lossArr = []
    for iterCount in range(max_iter):
        loss_new, sigArr = calLossFunc(X, y, w)
        lossArr.append(loss_new)
        if abs(loss_new - loss) <= tolerance:
            return w, lossArr
        updated = []
        for rowIndex in range(len(X)):
            updated.append(sigArr[rowIndex] - y[rowIndex])
        w = w - learning_rate*(np.matmul(np.transpose(X), updated))
        loss = loss_new
    return w, lossArr

def getPredictedOutputAndWeightsGradDecent(X, y, max_iter, learning_rate, tolerance):
    # appending one's array
    # X_final is of shape[[],[],...[]] i.e. n*(orig_no_of_feature_cols + 1), with 1st column of 1's
    X_final = np.append(np.ones((len(X), 1)), X, axis=1)        
    # bringing y in shape[[],[],...,[]] i.e. n*1
    y_final = []
    for ls in y:
        y_final.append([ls])            
    # initializing weights array w[[0], [0],...[0]] to size (orig_no_of_feature_cols + 1)*1
    w = np.zeros((len(X_final[0]), 1))  
    w, lossArr = getWeightsGradDecent(X_final, y_final, w, max_iter, learning_rate, tolerance)
    y_pred = []
    for rowIndex in range(len(X)):
        y_pred.append(calSigmoid(np.matmul(X_final[rowIndex],w)))  
    return y_pred, w, lossArr

def calZScore(dataset, meanArr, stdArr):
    for i in range(len(dataset[0]) - 1):
        for j in range(len(dataset)):
            dataset[j, i] = (dataset[j, i] - meanArr[i]) / stdArr[i]
    return dataset

def evaluate_algorithmGradDecent(dataset, n_folds, max_iter, learning_rate, tolerance):
    meanAcc_train = []
    meanRecall_train = []
    meanPre_train = []
    meanAcc_test = []
    meanRecall_test = []
    meanPre_test = []
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
                
        y_pred_train, w_train, lossArr = getPredictedOutputAndWeightsGradDecent(X_train, y_train, max_iter, learning_rate, tolerance)
        y_pred_train_final = []
        for y_data in y_pred_train:
            y_pred_train_final.append(0) if y_data < 0.5 else y_pred_train_final.append(1)
        
        X_test_final = np.append(np.ones((len(X_test), 1)), X_test, axis=1)
        y_pred_test = np.matmul(X_test_final, w_train)
        y_pred_test_final = []
        for y_data in y_pred_test:
            y_pred_test_final.append(0) if y_data < 0.5 else y_pred_test_final.append(1)
        
        meanAcc_train.append(accuracy_score(y_train, y_pred_train_final))
        meanRecall_train.append(recall_score(y_train, y_pred_train_final, average='micro'))
        meanPre_train.append(precision_score(y_train, y_pred_train_final, average='micro'))
        print("confusion_matrix(y_train, y_pred_train_final) = \n",confusion_matrix(y_train, y_pred_train_final))
 
        meanAcc_test.append(accuracy_score(y_test, y_pred_test_final))
        meanRecall_test.append(recall_score(y_test, y_pred_test_final, average='micro'))
        meanPre_test.append(precision_score(y_test, y_pred_test_final, average='micro'))
        print("confusion_matrix(y_test, y_pred_test_final) = \n",confusion_matrix(y_test, y_pred_test_final))
        print("-------------------------------------------------------------------------------------------------")
        
    print("Train Accuracy = ",meanAcc_train)
    print("Train Accuracy Mean = ",np.average(meanAcc_train))
    print("Train Accuracy standard deviation = ",np.std(meanAcc_train))
    print("Train Recall = ",meanRecall_train)
    print("Train Recall Mean = ",np.average(meanRecall_train))
    print("Train Recall standard deviation = ",np.std(meanRecall_train))
    print("Train Precision = ",meanPre_train)
    print("Train Precision Mean = ",np.average(meanPre_train))
    print("Train Precision standard deviation = ",np.std(meanPre_train))
    print("")
      
    print("Test Accuracy = ",meanAcc_test)
    print("Test Accuracy Mean = ",np.average(meanAcc_test))
    print("Test Accuracy standard deviation = ",np.std(meanAcc_test))
    print("Test Recall = ",meanRecall_test)
    print("Test Recall Mean = ",np.average(meanRecall_test))
    print("Test Recall standard deviation = ",np.std(meanRecall_test))
    print("Test Precision = ",meanPre_test)
    print("Test Precision Mean = ",np.average(meanPre_test))
    print("Test Precision standard deviation = ",np.std(meanPre_test))
    
    return meanAcc_train, lossArr

def main():
    
#     color = ['y', 'm', 'c', 'r', 'g', 'b', 'k', 'y', 'm', 'c', 'r', 'g', 'b', 'k']
#     markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'd', 'D']
#     tolerance = np.array([0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.002, 0.004, 0.006, 0.008, 0.01, 0.03, 0.05, 0.07, 0.09])
#     learningRate = np.array([0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.003, 0.005, 0.007, 0.009])
# 
#     lossArr = []
#     rmseArr = []
#     rmseArrAvg = []
#     lossArrMin = []
#     
#     lossArr2 = []
#     rmseArr2 = []
#     rmseArrAvg2 = []
#     lossArrMin2 = []
    
    print("Breast Cancer data: ")
    breastcancer = pd.read_csv('breastcancer.csv', sep=',', header=None)
    breastcancer.reindex(np.random.permutation(breastcancer.index)) 
    rmse_train, loss = evaluate_algorithmGradDecent(breastcancer.values, 10, 1000, 0.00001, 0.0003)

#     for t in tolerance:
#         print("*********************************************************************************")
#         print("tolerance = ",t)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(breastcancer.values, 10, 1000, 0.0004, t)
#         lossArr.append(loss)
#         rmseArr.append(rmse_train)
#         lossArrMin.append(np.min(loss))
#         rmseArrAvg.append(np.average(rmse_train))
#     print("\n")
#     for l in learningRate:
#         print("*********************************************************************************")
#         print("learning rate = ",l)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(breastcancer.values, 10, 1000, l, 0.004)
#         lossArr2.append(loss)
#         rmseArr2.append(rmse_train)
#         lossArrMin2.append(np.min(loss))
#         rmseArrAvg2.append(np.average(rmse_train))
#        
#     print("\n\n")        
     
    print("Pima Indian Diabetes data: ")
    diabetes = pd.read_csv('diabetes.csv', sep=',', header=None)
    diabetes.reindex(np.random.permutation(diabetes.index))
    diabetes_train_yacht_GradDecent = evaluate_algorithmGradDecent(diabetes.values, 10, 1000, 0.00001, 0.004)

#     for t in tolerance:
#         print("*********************************************************************************")
#         print("tolerance = ",t)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(diabetes.values, 10, 1000, 0.0004, t)
#         lossArr.append(loss)
#         rmseArr.append(rmse_train)
#         lossArrMin.append(np.min(loss))
#         rmseArrAvg.append(np.average(rmse_train))
#     print("\n")
#     for l in learningRate:
#         print("*********************************************************************************")
#         print("learning rate = ",l)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(diabetes.values, 10, 1000, l, 0.004)
#         lossArr2.append(loss)
#         rmseArr2.append(rmse_train)
#         lossArrMin2.append(np.min(loss))
#         rmseArrAvg2.append(np.average(rmse_train))
    
#     print("\n\n")    
#       
    print("Spambase data: ")
    spambase = pd.read_csv('spambase.csv', sep=',', header=None)
    spambase.reindex(np.random.permutation(spambase.index))
    spambase_train_yacht_GradDecent = evaluate_algorithmGradDecent(spambase.values, 10, 1000, 0.003, 0.006)

#     for t in tolerance:
#         print("*********************************************************************************")
#         print("tolerance = ",t)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(spambase.values, 10, 1000, 0.0004, t)
#         lossArr.append(loss)
#         rmseArr.append(rmse_train)
#         lossArrMin.append(np.min(loss))
#         rmseArrAvg.append(np.average(rmse_train))
#     print("\n")
#     for l in learningRate:
#         print("*********************************************************************************")
#         print("learning rate = ",l)
#         print("*********************************************************************************")
#         # evaluate_algorithm(dataset_as_ndarray, n_folds, max_iter, learning_rate, tolerance)
#         rmse_train, loss = evaluate_algorithmGradDecent(spambase.values, 10, 1000, l, 0.004)
#         lossArr2.append(loss)
#         rmseArr2.append(rmse_train)
#         lossArrMin2.append(np.min(loss))
#         rmseArrAvg2.append(np.average(rmse_train)) 
    
#     tolID = 0 
#     fig1 = plt.figure()
#     ax = fig1.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : RMSE VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('RMSE')
#     for t in tolerance:
#         ax.plot(np.full(len(rmseArr[tolID]), t), rmseArr[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='RMSE VS Tolerance')
#         tolID = tolID + 1        
#     tolID = 0
#     fig2 = plt.figure()             
#     ax = fig2.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : Avg. RMSE VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('Avg. RMSE')
#     ax.plot(tolerance, rmseArrAvg, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. RMSE VS Tolerance')
#     
#     tolID = 0 
#     fig3 = plt.figure()
#     ax = fig3.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : RMSE VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('RMSE')
#     for t in learningRate:
#         ax.plot(np.full(len(rmseArr2[tolID]), t), rmseArr2[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='RMSE VS learningRate')
#         tolID = tolID + 1        
#     tolID = 0
#     fig4 = plt.figure()             
#     ax = fig4.add_subplot(111)
#     ax.set_title('Deliverable 1.3.1 : Avg. RMSE VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('Avg. RMSE')
#     ax.plot(learningRate, rmseArrAvg2, ls='--', marker=markers[tolID], c=color[tolID], label='Avg. RMSE VS learningRate')
#         
#             
#     tolID = 0   
#     fig5 = plt.figure()           
#     ax = fig5.add_subplot(111)
#     ax.set_title('Deliverable 1.3.2 : Loss VS Iterations by varying tolerance')
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Loss')
#     for t in tolerance:
#         ax.plot(np.arange(1, len(lossArr[tolID])+1, 1), lossArr[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS Iterations by varying tolerance')
#         tolID = tolID + 1
#     tolID = 0   
#     fig6 = plt.figure()           
#     ax = fig6.add_subplot(111)
#     ax.set_title('Deliverable 1.3.2 : Loss VS Iterations by varying learningRate')
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Loss')
#     for t in learningRate:
#         ax.plot(np.arange(1, len(lossArr2[tolID])+1, 1), lossArr2[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS Iterations by varying learningRate')
#         tolID = tolID + 1
#         
#            
#     tolID = 0
#     fig7 = plt.figure()             
#     ax = fig7.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Loss VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('Loss')
#     for t in tolerance:
#         ax.plot(np.full(len(lossArr[tolID]), t), lossArr[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS Tolerance')
#         tolID = tolID + 1    
#     tolID = 0
#     fig8 = plt.figure()             
#     ax = fig8.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Min. Loss VS Tolerance')
#     ax.set_xlabel('Tolerance')
#     ax.set_ylabel('Min. Loss')
#     ax.plot(tolerance, lossArrMin, ls='--', marker=markers[tolID], c=color[tolID], label='Min. Loss VS Tolerance')
#     
#     tolID = 0
#     fig9 = plt.figure()             
#     ax = fig9.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Loss VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('Loss')
#     for t in learningRate:
#         ax.plot(np.full(len(lossArr2[tolID]), t), lossArr2[tolID], ls='--', marker=markers[tolID], c=color[tolID], label='Loss VS learningRate')
#         tolID = tolID + 1    
#     tolID = 0
#     fig10 = plt.figure()             
#     ax = fig10.add_subplot(111)
#     ax.set_title('Deliverable 1.3.3 : Min. Loss VS learningRate')
#     ax.set_xlabel('learningRate')
#     ax.set_ylabel('Min. Loss')
#     ax.plot(learningRate, lossArrMin2, ls='--', marker=markers[tolID], c=color[tolID], label='Min. Loss VS learningRate')
#         
#     plt.show()
        
if __name__ == "__main__": main()

