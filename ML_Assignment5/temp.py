'''
Created on Feb 7, 2018

@author: Mayuri
'''
import pandas as pd
import numpy as np
import matplotlib
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

def calGamma(X, pi, mean, var):
    gamma = np.zeros((len(X), len(pi)))
    for n in range(len(X)) :
        prob = []
        for k in range(len(pi)):
            norm = multivariate_normal.pdf(X[n], mean[k], var[k], allow_singular=True)
            prob.append(pi[k] * norm)
        den = np.sum(prob)
        for k in range(len(pi)):
            gamma[n, k] = prob[k] / den

    z = np.zeros((len(X), len(pi)))
    i = 0
    for row in gamma:
        max = np.argmax(row)
        z[i, max] = 1
        i = i + 1
    return gamma, z

def calMean(gamma, X, z):
    mean = []
    for k in range(len(z[0])):
        sum = 0
        for n in range(len(X)):
            sum = sum + gamma[n, k]*X[n]
        sum = sum / np.sum(gamma, axis=0)[k]
        mean.append(sum)
    return mean

def calVar(gamma, X, mean, z):
    var = []
    for k in range(len(z[0])):
        sum = 0
        for n in range(len(X)):
            sub = np.subtract(X[n], mean[k])
            test = np.outer(sub, sub)
            sum = sum + gamma[n, k] * test
        sum = sum / np.sum(gamma, axis=0)[k]
        var.append(sum)
    return var

def calPi(gamma):
    pi = []
    for k in range(len(gamma[0])):
        sum = 0
        for n in range(len(gamma)):
            sum = sum + gamma[n, k]
        sum = sum/len(gamma)
        pi.append(sum)
    return pi

def calZ(X, centroids):
    z = np.zeros((len(X), len(centroids)))
    distance = cdist(X, centroids, 'euclidean')
    i=0
    for row in distance:
        min = np.argmin(row)
        z[i, min] = 1
        i = i + 1
    return z

def calJ(z, X, centroids):
    J = 0
    p = 0
    for n in X:
        q = 0
        for k in centroids:
            J = J + z[p,q]*(np.sum(np.square(np.subtract(n, k))))
            q = q + 1
        p = p + 1
    return J

def calNMI(dataset, k, centroids):
    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    # cal class entrophy
    outputCategory, count = np.unique(dataset[:,-1], return_counts=True)
    classEntrophyFinal = 0
    for i in range(len(count)):
        prob = count[i]/len(dataset)
        if prob > 0:
            classEntrophyFinal = classEntrophyFinal - prob*(math.log(prob, 2))

    # cal cluster entrophy
    z = calZ(X, centroids)
    countOne = np.count_nonzero(z, axis=0)
    clusterEntrophyFinal = 0
    for i in range(len(countOne)):
        prob = countOne[i]/len(dataset)
        if prob > 0:
            clusterEntrophyFinal = clusterEntrophyFinal - prob*(math.log(prob, 2))

    # conditional entrophy of each class
    condClassEntrophy = 0
    for i in range(len(z[0])):
        listY = []
        m = 0
        for j in z[:,i]:
            if (j != 0):
                listY.append(y[m])
            m = m + 1
        outputY, countY = np.unique(listY, return_counts=True)
        total = np.sum(countY)
        outputYFinal = 0
        for b in range(len(countY)):
            prob = countY[b] / total
            if prob > 0:
                outputYFinal = outputYFinal + prob * (math.log(prob, 2))
        condClassEntrophy = condClassEntrophy + (-1 * outputYFinal * (countOne[i]/len(dataset)))

    # cal mutual info
    mi = classEntrophyFinal - condClassEntrophy
    nmi = (2 * mi)/(classEntrophyFinal + clusterEntrophyFinal)
    return nmi

def evaluate_algorithmGMM(dataset, minK, maxK, tol, maxIter):

    X = dataset[:, 0:-1]
    J = []
    nmi = []
    centroidsList = []

    for k in range(minK, maxK + 1):
        print("k = ",k)
        kmean = KMeans(n_clusters=k, random_state=0).fit(X)
        var = []
        centroids = kmean.cluster_centers_
        meanLabel = kmean.labels_
        for clusterInx in range(k):
            varCluster = np.zeros((len(X[0]), len(X[0])))
            for n in range(len(X)):
                if meanLabel[n] == clusterInx:
                    sub = np.subtract(X[n], centroids[clusterInx])
                    test = np.outer(sub, sub)
                    varCluster = varCluster + test
            var.append(varCluster)

        z = calZ(X, centroids)
        countOne = np.count_nonzero(z, axis=0)
        pi = []
        for i in range(len(countOne)):
            pi.append(countOne[i] / len(dataset))

        gamma, z = calGamma(X, pi, centroids, var)
        mean = calMean(gamma, X, z)
        var = calVar(gamma, X, mean, z)
        pi = calPi(gamma)

        stoppingCriteria = 0
        stoppingCriteriaNew = 0
        iter = 0
        while iter<=maxIter:
            gamma, z = calGamma(X, pi, mean, var)
            mean = calMean(gamma, X, z)
            var = calVar(gamma, X, mean, z)
            pi = calPi(gamma)
            for p in range(len(X)):
                temp = 0
                for q in range(k):
                    norm = multivariate_normal.pdf(X[p], mean[q], var[q], allow_singular=True)
                    temp = temp + pi[q]*norm
                stoppingCriteria = stoppingCriteria + math.log(temp, 2)
            if (stoppingCriteria == stoppingCriteriaNew or abs(stoppingCriteriaNew-stoppingCriteria)<=tol):
                break;
            stoppingCriteriaNew = stoppingCriteria
            iter = iter + 1

        print("calJ(z, X, mean) = ",calJ(z, X, mean))
        J.append(calJ(z, X, mean))
        print("calNMI(dataset, k, mean) = ",calNMI(dataset, k, mean))
        nmi.append(calNMI(dataset, k, mean))
        centroidsList.append(centroids)
    return J, nmi

def main():

    print("dermatologyData: ")
    dermatologyData = pd.read_csv('dermatologyData.csv', sep=',', header=None)
    dermatologyData.reindex(np.random.permutation(dermatologyData.index))
    minK = 2
    maxK = 10
    tol = 500
    maxIter = 500
    labelCount = 7
    title = "dermatologyData"
    # myThread(threadID, counter, dataset, minK, maxK, maxIter, tol, labelCount)
    # thread1 = myThread(1, 1, dermatologyData.values, 2, 10, 500, 500, 7, "dermatologyData")
    # thread1.start()
    # threads.append(thread1)
    # print("")

    # print("vowelsData: ")
    # vowelsData = pd.read_csv('vowelsData.csv', sep=',', header=None)
    # vowelsData.reindex(np.random.permutation(vowelsData.index))
    # # myThread(threadID, counter, dataset, minK, maxK, maxIter, tol, labelCount)
    # thread2 = myThread(2, 2, vowelsData.values, 2, 20, 1000, 300, 11, "vowelsData")
    # thread2.start()
    # threads.append(thread2)
    # print("")
    #
    # print("glassData: ")
    # glassData = pd.read_csv('glassData.csv', sep=',', header=None)
    # glassData.reindex(np.random.permutation(glassData.index))
    # # myThread(threadID, counter, dataset, minK, maxK, maxIter, tol, labelCount)
    # thread3 = myThread(3, 3, glassData.values, 2, 20, 1000, 300, 6, "glassData")
    # thread3.start()
    # threads.append(thread3)
    # print("")
    #
    # print("ecoliData: ")
    # ecoliData = pd.read_csv('ecoliData.csv', sep=',', header=None)
    # ecoliData.reindex(np.random.permutation(ecoliData.index))
    # # myThread(threadID, counter, dataset, minK, maxK, maxIter, tol, labelCount)
    # thread4 = myThread(4, 4, ecoliData.values, 2, 20, 1000, 300, 5, "ecoliData")
    # thread4.start()
    # threads.append(thread4)
    # print("")
    #
    # print("yeastData: ")
    # yeastData = pd.read_csv('yeastData.csv', sep=',', header=None)
    # yeastData.reindex(np.random.permutation(yeastData.index))
    # # myThread(threadID, counter, dataset, minK, maxK, maxIter, tol, labelCount)
    # thread5 = myThread(5, 5, yeastData.values, 2, 20, 1000, 300, 9, "yeastData")
    # thread5.start()
    # threads.append(thread5)
    # print("")
    #
    # print("soybeanData: ")
    # soybeanData = pd.read_csv('soybeanData.csv', sep=',', header=None)
    # soybeanData.reindex(np.random.permutation(soybeanData.index))
    # # myThread(threadID, counter, dataset, minK, maxK, maxIter, tol, labelCount)
    # thread6 = myThread(6, 6, soybeanData.values, 2, 20, 1000, 300, 15, "soybeanData")
    # thread6.start()
    # threads.append(thread6)
    # print("")

    # for t in threads:
    #     t.join()
    # print("Exiting Main Thread")

    sse1, nmi1 = evaluate_algorithmGMM(dermatologyData.values, minK, maxK, tol, maxIter)
    print("sse = ", sse1)
    print("nmi = ", nmi1)
    # sse.append(sse1)
    # nmi.append(nmi1)
    res = np.argmax(nmi1)
    print("optimal k = ", res + 2)
    print("best NMI = ", nmi1[res])
    print("optimal SSE = ", sse1[res])
    print("On setting the number of clusters equal to the number of classes:")
    print("k = ", labelCount)
    print("NMI = ", nmi1[labelCount-2])
    print("SSE = ", sse1[labelCount-2])
    print("")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.set_title(title)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Objective function / SSE')
    ax1.plot(np.arange(minK, maxK+1, 1), sse1, ls='--', marker='o', c='y', label='SSE')

    ax2 = fig1.add_subplot(122)
    ax2.set_title(title)
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('NMI')
    ax2.plot(np.arange(minK, maxK+1, 1), nmi1, ls='--', marker='v', c='m', label='NMI')

    plt.show()


if __name__ == "__main__": main()