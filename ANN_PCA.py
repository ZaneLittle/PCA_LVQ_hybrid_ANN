""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Principal Component Analysis using the Artificial Neural Network Approach"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd

class ANN_PCA(object):
  def __init__(self, inputs, outputs):
    self.weights = [np.random.random_sample(inputs) for i in range(outputs)] 


  def generate_single_output(self, x):
    out = []
    for node in self.weights:
      out.append(np.dot(x, node))
    return out


  def convert_data(self, dataset):
    converted_data = []
    for data in dataset:
      converted_data.append(self.generate_single_output(data))
    return converted_data
    

  def train(self, train_x, lr=0.1, epochs=100):
    # Normalize data
    x_df = pd.DataFrame(data=train_x)
    x_df = x_df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    test_x = x_df.values

    for epoch in range(epochs):
      for xi in test_x:
        Y = np.array(self.generate_single_output(xi))

        # Computing outer dot product and update weights
        # weight += lr * YX - KW, where K = Y Y.T
        KW =  np.multiply(np.dot(Y, Y.T),self.weights)
        self.weights += lr * np.outer(Y, xi) - KW