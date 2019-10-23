import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                      Helper Functions                                    """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def convert_results(dataset):
  """   Define result set for given training set as 0, 1, 2   """
  for i in range(len(dataset)):
    if dataset[i][-1] == 'Iris-setosa':
      dataset[i][-1] = 0
    elif dataset[i][-1] == 'Iris-versicolor':
      dataset[i][-1] = 1
    elif dataset[i][-1] == 'Iris-virginica':
      dataset[i][-1] = 2

  return dataset


def convert_to_names(result_set):
  """   Converts array of 0,1,2 values to species names  """
  names = []
  for result in result_set:
    if int(result) == 0:
      names.append('Iris-setosa')
    elif int(result) == 1:
      names.append('Iris-versicolor')
    elif int(result) == 2:
      names.append('Iris-virginica')

  return names


def convert_to_df(arr):
  """ convert array to dataframe  """
  c = ['sepal length', 'sepal width', 'petal length', 'petal width', 'classification']
  df = pd.DataFrame(data=arr, columns=c)
  
  # lengths and widths to floats
  df['sepal length'] = df['sepal length'].astype(float)
  df['sepal width'] = df['sepal width'].astype(float)
  df['petal length'] = df['petal length'].astype(float)
  df['petal width'] = df['petal width'].astype(float)

  return df


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                         IO Functions                                     """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def read_files(data_files):
  """
  Read Files into notebook
    dataFiles: list of files to read in
  """
  for i in range(len(data_files)):
    uploaded = files.upload()

    for fn in uploaded.keys():
      print('uploaded: "{name}" ({length} bytes)'.format(
        name=fn, length=len(uploaded[fn])))
      

def read_data(filename):
  """ Read training data into test and train vars """
  with open(filename, 'r') as f: 
   data = f.read()

  # convert data
  data_arr = data.splitlines()
  feature_arr = []
  for data_point in data_arr:
    feature = data_point.split(',')
    feature_arr.append(feature)
  
  convert_results(feature_arr)

  return np.array(feature_arr).astype(float)


def writeResults(resultSet):
  with open('results.txt', 'w') as f:
    f.write(resultSet.to_string())



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                 Visualization Functions                                  """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def plotErrs(errList, title):
  """
  Plot graph of errors with given title
  """
  plt.plot(range(1, len(errList) + 1), errList, marker='o')
  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel('Uncertainty (%)')
  plt.show()


def plot_confusion_matrix(y_true, 
                        y_pred, 
                        normalize=False,
                        title='Results Confusion matrix',
                        cmap=plt.cm.Blues):
  """
  Plots a 3d confusion matrix
    y_true is the list true values
    y_pred is the list of actual values
  """

  classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

  cm = confusion_matrix(y_true, y_pred, labels=classes)
  
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
