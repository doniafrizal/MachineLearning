# used for manipulating directory path
import os
import regresion as lr
import pandas as pd


# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Read comma separated data

data = pd.read_csv(os.path.join('data', 'ex1data1.txt'), delimiter=',', header=None)

data.head()

data.describe()

data_feature_label = data.values

total_examples = data_feature_label[:, 0].size  # number of training examples

# Setup Feature and Label.
feature = np.append(np.ones((total_examples, 1)), data_feature_label[:, 0].reshape(total_examples, 1), axis=1)
label = data_feature_label[:, 1].reshape(total_examples, 1)

# Initialize theta to zero values
theta = np.zeros((2, 1))

cost = lr.compute_cost(feature, label, theta)
print('With theta = [0, 0] \nCost computed = %.2f' % cost)
print('Expected cost value (approximately) 32.07\n')

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, cost_history = lr.gradient_descent(feature, label, theta, alpha, iterations)
print('Theta found by gradient descent: ' + str(round(theta[0, 0], 2)) + ' , ' + str(round(theta[1, 0], 2)))

lr.plot_check_cost(cost_history)  # Plot cost function to verify minimum cost achieved.

# plot the linear fit
lr.plot_data(feature, label, 4, 25, -5, 25, 'Population in 10,000s', 'Profit in $10,000s')
pyplot.plot(feature[:, 1], np.dot(feature, theta), '-')
pyplot.legend(['Training data', 'Linear regression']);

lr.plot_cost(feature, label, theta)  # Plot cost to verify global minimum value achievement

# Predict values for population sizes of 35,000 and 70,000
predict1 = lr.predict(np.array([1, 3.5]), theta) * 10000
print('For population = 35,000, we predict a profit of : ' + str(round(predict1, 2)))

predict2 = lr.predict(np.array([1, 7]), theta) * 10000
print('For population = 70,000, we predict a profit of : ' + str(round(predict2, 2)))
