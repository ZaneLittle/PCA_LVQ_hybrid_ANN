""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Learning Vector Quantization Algorithm and PCA implementation            """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from random import randrange
from math import sqrt
 
def euclidean_distance(row1, row2):
	""" Calculates the Euclidean distance between two vectors	"""
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 

def get_best_matching_unit(codebooks, test_row):
	""" Return the closest cluster	"""
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]
 

def random_codebook(dataset):
	""" Returns a random codebook picked from a dataset	"""
	n_records = len(dataset)
	n_features = len(dataset[0])
	codebook = [dataset[randrange(n_records)][i] for i in range(n_features)]
	return codebook
 

def train_codebooks(dataset, clusters, lr, epochs):
	"""
	Train a set of codebook vectors
	Returns trained codebooks
	"""
	# Pick random codebooks
	codebooks = [random_codebook(dataset) for i in range(clusters)]
	# Re-label each codebook so we have one for each cluster
	for i in range(clusters):
		codebooks[i][-1] = i

	for epoch in range(epochs):
		for row in dataset:

			bmu = get_best_matching_unit(codebooks, row)

			for i in range(len(row)-1):
				error = row[i] - bmu[i]

				# If classification is correct, pull bmu closer, else push away
				if bmu[-1] == row[-1]:
					bmu[i] += lr * error
				else:
					bmu[i] -= lr * error
	return codebooks 


def test_codebooks(codebooks, test):
	""" 
	Test a given set of codebooks on dataset
	Returns predictions and prints accuracy
	"""
	err = 0
	results = []
	for i in test: 
		bmu = get_best_matching_unit(codebooks, i)
		results.append(bmu[-1])
		if i[-1] != bmu[-1]:
			err += 1

	acc = 100 - (err / len(test) * 100) 
	print("Accuracy in test: ", acc, "%")
 
	return results

def stat_PCA(X):
	"""
	Performs Principle Component Analysis on a dataset using the statistical approach
		X: dataset of inputs, stripped of labels 
	Returns the resulting eigen vectors and eigen values
	"""
	M = np.mean(dataset.T, axis=1)
	# center columns by subtracting column means
	C = dataset - M
	# calculate covariance matrix of centered matrix
	V = cov(C.T)
	# eigendecomposition of covariance matrix
	return eig(V)

def LVQ(train, test):
	"""	Implementation 1: simple LVQ Network	"""
	# Define algorithm params
	lr = 0.1
	n_epochs = 50
	clusters = 3

	# Train codebooks
	codebooks = train_codebooks(train, clusters, lr, n_epochs)

	# Test codebooks
	results = convert_to_names(test_codebooks(codebooks, test))
	actual = convert_to_names(pd.DataFrame(data=test).iloc[:, -1].values)
	
	print("Final codebook: ", codebooks)

	plot_confusion_matrix(actual, results)


def hybrid_PCA_LVQ(train, test):
	"""	Implementation 2: PCA LVQ hybrid Network	"""
	# Strip labels from data
	train_df = convert_to_df(train)
	test_df = convert_to_df(test)

	train_x = train_df.iloc[0:, [0,1,2,3]].values
	test_x = test_df.iloc[0:, [0,1,2,3]].values

	# Define PCA net
	(_, inputs) = train_x.shape
	outputs = 3
	PCA = ANN_PCA(inputs, outputs)
	
	# Train the PCA net
	PCA.train(train_x)
	
	# Perform feature extraction on training and testing data
	extracted_train = PCA.convert_data(train_x)
	extracted_test = PCA.convert_data(test_x)
 
	# Re-add labels
	extracted_train_df = pd.DataFrame(data=extracted_train)
	extracted_train_df.insert(loc=3, column='classification', value=train_df['classification'])
	new_train=extracted_train_df.values

	extracted_test_df = pd.DataFrame(data=extracted_test)
	extracted_test_df.insert(loc=3, column='classification', value=test_df['classification'])
	new_test=extracted_test_df.values
	
	# Perform LVQ on feature extracted data
	LVQ(new_train, new_test)