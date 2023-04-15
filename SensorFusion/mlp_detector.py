from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np


def MLP():
  # Import dataset
  dataset = pd.read_csv("D:\\downloads\\YOLO v7\\thesis_v3\\thesis-beta-1\\mlp.csv")

  # Seperate the unique column
  X = dataset[['yolov_pred', 'capacitive_pred', 'inductive_pred']]
  y = dataset['result_pred']

  # Splitting of training and testing
  
  X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)


  # Traning the model
  
  mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)

  mlp.fit(X_train, y_train)
  # print(X_train)

  return mlp

mlp = MLP()