"""
A module to test how effective the neural network implementation is by using a real housing dataset and predicting
housing prices based on that
Author: Naoroj Farhan
Date: Thursday, May 23, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from losses import *
from functions import *
from neural_network import NN
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Housing.csv')
train_data, test_data = train_test_split(data, test_size=0.2)
train_data = pd.DataFrame(train_data).to_numpy()
test_data = pd.DataFrame(test_data).to_numpy()

x_train = np.column_stack((train_data[:,1], train_data[:,2], train_data[:,3], train_data[:,4], train_data[:,5],train_data[:,6], train_data[:,7], train_data[:,8], train_data[:,9],train_data[:,10], train_data[:,11]))
y_train = train_data[:, 0]
x_test = np.column_stack((test_data[:,1], test_data[:,2], test_data[:,3], test_data[:,4], test_data[:,5],test_data[:,6], test_data[:,7], test_data[:,8], test_data[:,9],test_data[:,10], test_data[:,11]))
y_test = test_data[:, 0]

# Using one hot encoding to change categorical data into numerical data
for i in range (x_train.shape[0]):
    for j in range(x_train.shape[1]):
        if x_train[i][j] == 'yes':
            x_train[i][j] = 1
        elif x_train[i][j] == 'no':
            x_train[i][j] = 0
            
for i in range (x_test.shape[0]):
    for j in range(x_test.shape[1]):
        if x_test[i][j] == 'yes':
            x_test[i][j] = 1
        elif x_test[i][j] == 'no':
            x_test[i][j] = 0

# Taking the transpose. x is a data matrix where each column is an individual example
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

y_train = np.reshape(y_train, (1, y_train.shape[0]))
y_test = np.reshape(y_test, (1, y_test.shape[0]))



# converting all the data values to float types so numpy methods work 
x_train = x_train.astype(float)
y_train = y_train.astype(float)
x_test = x_test.astype(float)
y_test = y_test.astype(float)


predictor = NN([('input', 11), ('relu', 1)], 'MeanSquaredError')

iters = 100
costs = predictor.gradient_descent(iters, 0.1, x_train, y_train)

plt.plot(np.arange(iters), costs)
plt.title("Learning Curve for Neural Network")
plt.ylabel("Cost function")
plt.xlabel("iterations")
plt.show()

# I'm too lazy atp to make good generalized code, I'mma hard code for mean squared error
mse = 0
loss_function = MeanSquaredError()
mse = loss_function.evaluate(predictor.forward_propagation(x_test), y_test)
print(mse / y_test.shape[1])

# You will notice mean squared error is around 2x10^12 so the mean absolute error is 500,000 dollars ish. Turns out vanilla neural networks
# aren't so good at predicting housing prices?