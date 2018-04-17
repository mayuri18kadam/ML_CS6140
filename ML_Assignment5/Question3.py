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
        classEntrophyFinal = classEntrophyFinal - prob*(math.log(prob, 2))

    # cal cluster entrophy
    z = calZ(X, centroids)
    countOne = np.count_nonzero(z, axis=0)
    clusterEntrophyFinal = 0
    for i in range(len(countOne)):
        prob = countOne[i]/len(dataset)
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
            outputYFinal = outputYFinal + prob * (math.log(prob, 2))
        condClassEntrophy = condClassEntrophy + (-1 * outputYFinal * (countOne[i]/len(dataset)))

    # cal mutual info
    mi = classEntrophyFinal - condClassEntrophy
    nmi = (2 * mi)/(classEntrophyFinal - clusterEntrophyFinal)
    return nmi

def evaluate_algorithm(dataset, maxK):
    X = dataset[:, 0:-1]
    y = dataset[:, -1]
    J = []
    nmi = []
    centroidsList = []
    for k in range(2, maxK+1):
        centroids = X[0:k, :]
        centroidsNew = np.zeros_like(centroids)
        while True:
            z = calZ(X, centroids)
            countOne = np.count_nonzero(z, axis=0)
            temp = np.matmul(z.T, X)
            for i in range(len(centroidsNew)):
                for j in range(len(centroidsNew[0])):
                    centroidsNew[i,j] = temp[i,j] / countOne[i]
            if (np.array_equal(centroids, centroidsNew)):
                break;
            centroids = centroidsNew
        J.append(calJ(z, X, centroids))
        nmi.append(calNMI(dataset, k, centroids))
        centroidsList.append(centroids)
    # print(J)
    # print(nmi)
    return J, nmi

def main():
    print("dermatologyData: ")
    dermatologyData = pd.read_csv('dermatologyData.csv', sep=',', header=None)
    dermatologyData.reindex(np.random.permutation(dermatologyData.index))
    # evaluate_algorithm(dataset_as_ndarray, noOfClusters)
    J_derm , nmiDerm = evaluate_algorithm(dermatologyData.values, 20)
    res = np.argmax(nmiDerm)
    print("optimal k = ",res+2)
    print("best NMI = ",nmiDerm[res])
    print("optimal SSE = ", J_derm[res])
    print("On setting the number of clusters equal to the number of classes:")
    print("k = 7")
    print("NMI = ", nmiDerm[5])
    print("SSE = ", J_derm[5])
    print("")

    print("vowelsData: ")
    vowelsData = pd.read_csv('vowelsData.csv', sep=',', header=None)
    vowelsData.reindex(np.random.permutation(vowelsData.index))
    # evaluate_algorithm(dataset_as_ndarray, noOfClusterse)
    J_vowels , nmiVowels = evaluate_algorithm(vowelsData.values, 30)
    res2 = np.argmax(nmiVowels)
    print("optimal k = ", res2+2)
    print("best NMI = ", nmiVowels[res2])
    print("optimal SSE = ", J_vowels[res2])
    print("On setting the number of clusters equal to the number of classes:")
    print("k = 11")
    print("NMI = ", nmiVowels[9])
    print("SSE = ", J_vowels[9])
    print("")

    print("glassData: ")
    glassData = pd.read_csv('glassData.csv', sep=',', header=None)
    glassData.reindex(np.random.permutation(glassData.index))
    # evaluate_algorithm(dataset_as_ndarray, noOfClusters)
    J_glass , nmiGlass = evaluate_algorithm(glassData.values, 20)
    res3 = np.argmax(nmiGlass)
    print("optimal k = ", res3+2)
    print("best NMI = ", nmiGlass[res3])
    print("optimal SSE = ", J_glass[res3])
    print("On setting the number of clusters equal to the number of classes:")
    print("k = 6")
    print("NMI = ", nmiGlass[4])
    print("SSE = ", J_glass[4])
    print("")

    print("ecoliData: ")
    ecoliData = pd.read_csv('ecoliData.csv', sep=',', header=None)
    ecoliData.reindex(np.random.permutation(ecoliData.index))
    # evaluate_algorithm(dataset_as_ndarray, noOfClusters)
    J_ecoli , nmiEcoli = evaluate_algorithm(ecoliData.values, 20)
    res4 = np.argmax(nmiEcoli)
    print("optimal k = ", res4+2)
    print("best NMI = ", nmiEcoli[res4])
    print("optimal SSE = ", J_ecoli[res4])
    print("On setting the number of clusters equal to the number of classes:")
    print("k = 5")
    print("NMI = ", nmiEcoli[3])
    print("SSE = ", J_ecoli[3])
    print("")

    print("yeastData: ")
    yeastData = pd.read_csv('yeastData.csv', sep=',', header=None)
    yeastData.reindex(np.random.permutation(yeastData.index))
    # evaluate_algorithm(dataset_as_ndarray, noOfClusters)
    J_yeast , nmiYeast = evaluate_algorithm(yeastData.values, 20)
    res5 = np.argmax(nmiYeast)
    print("optimal k = ", res5+2)
    print("best NMI = ", nmiYeast[res5])
    print("optimal SSE = ", J_yeast[res5])
    print("On setting the number of clusters equal to the number of classes:")
    print("k = 9")
    print("NMI = ", nmiYeast[7])
    print("SSE = ", J_yeast[7])
    print("")

    print("soybeanData: ")
    soybeanData = pd.read_csv('soybeanData.csv', sep=',', header=None)
    soybeanData.reindex(np.random.permutation(soybeanData.index))
    # evaluate_algorithm(dataset_as_ndarray, noOfClusters)
    J_soya , nmiSoya = evaluate_algorithm(soybeanData.values, 20)
    res6 = np.argmax(nmiSoya)
    print("optimal k = ", res6+2)
    print("best NMI = ", nmiSoya[res6])
    print("optimal SSE = ", J_soya[res6])
    print("On setting the number of clusters equal to the number of classes:")
    print("k = 15")
    print("NMI = ", nmiSoya[13])
    print("SSE = ", J_soya[13])
    print("")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.set_title('dermatologyData')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Objective function / SSE')
    ax1.plot(np.arange(1, 20, 1), J_derm, ls='--', marker='o', c='y', label='drop in error')

    ax2 = fig1.add_subplot(122)
    ax2.set_title('dermatologyData')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('NMI')
    ax2.plot(np.arange(1, 20, 1), nmiDerm, ls='--', marker='o', c='y', label='NMI')

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(121)
    ax1.set_title('vowelsData')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Objective function / SSE')
    ax1.plot(np.arange(1, 30, 1), J_vowels, ls='--', marker='v', c='m', label='drop in error')

    ax2 = fig2.add_subplot(122)
    ax2.set_title('vowelsData')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('NMI')
    ax2.plot(np.arange(1, 30, 1), nmiVowels, ls='--', marker='o', c='y', label='NMI')

    fig3 = plt.figure()
    ax1 = fig3.add_subplot(121)
    ax1.set_title('glassData')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Objective function / SSE')
    ax1.plot(np.arange(1, 20, 1), J_glass, ls='--', marker='^', c='c', label='drop in error')

    ax2 = fig3.add_subplot(122)
    ax2.set_title('glassData')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('NMI')
    ax2.plot(np.arange(1, 20, 1), nmiGlass, ls='--', marker='o', c='y', label='NMI')

    fig4 = plt.figure()
    ax1 = fig4.add_subplot(121)
    ax1.set_title('ecoliData')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Objective function / SSE')
    ax1.plot(np.arange(1, 20, 1), J_ecoli, ls='--', marker='<', c='r', label='drop in error')

    ax2 = fig4.add_subplot(122)
    ax2.set_title('ecoliData')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('NMI')
    ax2.plot(np.arange(1, 20, 1), nmiEcoli, ls='--', marker='o', c='y', label='NMI')

    fig5 = plt.figure()
    ax1 = fig5.add_subplot(121)
    ax1.set_title('yeastData')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Objective function / SSE')
    ax1.plot(np.arange(1, 20, 1), J_yeast, ls='--', marker='>', c='g', label='drop in error')

    ax2 = fig5.add_subplot(122)
    ax2.set_title('yeastData')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('NMI')
    ax2.plot(np.arange(1, 20, 1), nmiYeast, ls='--', marker='o', c='y', label='NMI')

    fig6 = plt.figure()
    ax1 = fig6.add_subplot(121)
    ax1.set_title('soybeanData')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Objective function / SSE')
    ax1.plot(np.arange(1, 20, 1), J_soya, ls='--', marker='s', c='b', label='drop in error')

    ax2 = fig6.add_subplot(122)
    ax2.set_title('soybeanData')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('NMI')
    ax2.plot(np.arange(1, 20, 1), nmiSoya, ls='--', marker='o', c='y', label='NMI')

    plt.show()


if __name__ == "__main__": main()