
import numpy as np
import pandas as pd
import matplotlib
import math
matplotlib.use('TkAgg')



def evaluate_algorithm(dataset):
    return 0

def main():
    print("Iris data:")
    iris_data = pd.read_csv('iris_normalized.csv', sep=',', header=None)
    T = np.array([100, 200, 300, 400, 500])
    evaluate_algorithm(iris_data.values, T)

if __name__ == "__main__": main()