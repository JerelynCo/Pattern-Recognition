import pandas as pd
import numpy as np
import os
import random
import math
import json

# Naive bayes src:
# http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/


def collate(saved_dir, data_fn):
    df_arr = []
    for file in os.listdir(saved_dir):
        df_arr.append(pd.read_csv(saved_dir + file, names=None))
    data = pd.concat(df_arr).reset_index(drop=True)
    data.to_csv(data_fn, index=False)
    print("All csv files collated.")


def generateFiles():
    # directories
    saved_dir = "../saved/"  # directory where all uncollated files are
    data_fn = "normalized_all.csv"

    # uncomment line below if no "data.csv"
    if not os.path.exists(data_fn):
        collate(saved_dir, data_fn)

    data = pd.read_csv(data_fn)
    train, test = splitDataset(data)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    print("Train and test data saved.")


def splitDataset(df):
    mask = np.random.rand(len(df)) < 0.8
    train = df[mask]
    test = df[~mask]
    print("Split data with {0} rows into train with {1} rows and test with {2} rows".format(
        df.shape[0], train.shape[0], test.shape[0]))
    return [train, test]


def separateByClass(df):
    print("Train data with {0} rows of non-objects and {1} rows of objects".format(
        df[df['class'] == 0].shape[0], df[df['class'] == 1].shape[0]))
    return (df[df['class'] == 0], df[df['class'] == 1])

# Separates features and labels


def separateXandY(df):
    n_cols = df.shape[1] - 1
    X = df.iloc[:, 0:n_cols]
    y = df.iloc[:, n_cols]
    print(
        "Separated features and labels. X = features, y = labels/classifications")
    return (X, y)


def getStats(df):
    # get all columns except 'class'
    print("Mean and standard deviation returned.")
    return (df.mean().values, df.std().values)


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    generateFiles()

    train = pd.read_csv("train.csv")
    train_0, train_1 = separateByClass(train)

    X_train_0, y_train_0 = separateXandY(train_0)
    X_train_1, y_train_1 = separateXandY(train_1)

    train_stats = [getStats(X_train_0), getStats(X_train_1)]

    stats_arr = [[], []]
    for i in range(2):
        for stats in zip(train_stats[i][0], train_stats[i][1]):
            stats_arr[i].append(stats)

    train_summary = {0: stats_arr[0], 1: stats_arr[1]}

    test = pd.read_csv("test.csv")
    X_test, y_test = separateXandY(test)

    # dumping of train data summary
    with open('train_summary.json', 'w') as f:
        f.write(json.dumps(train_summary))

    print(getAccuracy(y_test, getPredictions(train_summary, X_test.values)))

if __name__ == '__main__':
    main()
# to get the 'model', export the train_summary and copy the getPrediction
# method to the opencv file
