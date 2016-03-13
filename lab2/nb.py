import pandas as pd
import os
import random
import math

def collate(data_dir):
	df_arr = []
	for file in os.listdir(saved_dir):
		df_arr.append(pd.read_csv(data_dir + file, names=None))
	data = pd.concat(df_arr).reset_index(drop=True)
	data.to_csv(data_dir + "all.csv", index=False)

def splitDataset(df):
	mask = np.random.rand(len(df)) < 0.8
	train = df[mask]
	test = df[~mask]
	print("Split data with {0} rows into train with {1} rows and test with {2} rows".format(len(df), len(train), len(test)))
	return [train, test]

def separateByClass(df):
	return (df[df['class'] == 0], df[df['class'] == 1])

def getStats(df):
	# get all columns except 'class'
	n_cols = df.shape[1]-1
	df = df.iloc[:,0:n_cols]
	return (df.mean().values, df.std().values)


def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


data_dir = "saved/"
collate(data_dir)

data = pd.read_csv(data_dir + "all.csv")
train, test = splitDataset(data)

train_0, train_1 = separateByClass(train)


train_stats = [getStats(train_0), getStats(train_1)]

stats_arr = [[], []]
for i in range(2):
	for stats in zip(train_stats[i][0], train_stats[i][1]):
		stats_arr[i].append(stats)

train_dict = {0:stats_arr[0], 1:stats_arr[1]}

