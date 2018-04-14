import pandas as pd
import numpy as np

def main():
    iris = pd.read_csv('iris.csv', header=None)
    spambase = pd.read_csv('spambase.csv', header=None)
    housing = pd.read_csv('housing.csv', header=None)

    iris_max = np.array(iris.max())
    iris_min = np.array(iris.min())
    for i in range(len(iris.columns) - 1):
        for j in range(iris.shape[0]):
            iris.iloc[j, i] = (iris.iloc[j, i] - iris_min[i]) / (iris_max[i] - iris_min[i])
    iris.reindex(np.random.permutation(iris.index)).to_csv('iris_normalized.csv', index=False, header=None)

    spambase_max = np.array(spambase.max())
    spambase_min = np.array(spambase.min())
    for i in range(len(spambase.columns) - 1):
        for j in range(spambase.shape[0]):
            spambase.iloc[j, i] = (spambase.iloc[j, i] - spambase_min[i]) / (spambase_max[i] - spambase_min[i])
    spambase.reindex(np.random.permutation(spambase.index)).to_csv('spambase_normalized.csv', index=False, header=None)

    housing_max = np.array(housing.max())
    housing_min = np.array(housing.min())
    for i in range(len(housing.columns) - 1):
        for j in range(housing.shape[0]):
            housing.iloc[j, i] = (housing.iloc[j, i] - housing_min[i]) / (housing_max[i] - housing_min[i])
    housing.reindex(np.random.permutation(housing.index)).to_csv('housing_normalized.csv', index=False, header=None)

if __name__ == "__main__": main()

