""" Simple script to compare LVQ and LVQ with PCA Feature extraction for pre-processing """
from utils import *
from LVQ import *

# Read data
train = read_data('iris_train.txt')
test = read_data('iris_test.txt')

# Implementation 1: simple LVQ Network
LVQ(train, test)

print("\n\n")

# Implementation 2: PCA LVQ hybrid Network
hybrid_PCA_LVQ(train, test) 
