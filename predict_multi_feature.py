# used for manipulating directory path
import os
import regresion as lr
import pandas as pd

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Read comma separated data

data = pd.read_csv(os.path.join('data', 'ex1data2.txt'), delimiter=',', header=None)

data.head()

data.describe()

data_feature_label = data.values

total_examples = len(data_feature_label[:, -1])  # number of training examples

# Setup Feature and Label.
feature = data_feature_label[:, 0:2].reshape(total_examples, 2)
label = data_feature_label[:, -1].reshape(total_examples, 1)

# call featureNormalize on the loaded data
feature_norm, mean, sigma = lr.feature_normalize(feature)

print('Computed mean:', mean)
print('Computed standard deviation:', sigma)

# Append matrix m x 1 of one
feature_norm = np.append(np.ones((total_examples, 1)), feature_norm, axis=1)

# Initialize theta to zero values
theta = np.zeros((3, 1))
cost = lr.compute_cost(feature_norm, label, theta)  # Calculate initial cost

# some gradient descent settings
iterations = 1400
alpha = 0.1

theta, cost_history = lr.gradient_descent(feature_norm, label, theta, alpha, iterations)
print('Theta found by gradient descent: ' + str(round(theta[0, 0], 2)) + ' , ' + str(round(theta[1, 0], 2)) + ' , '
      + str(round(theta[2, 0], 2)))

lr.plot_check_cost(cost_history)  # Plot cost function to verify minimum cost achieved.

# Predict values using optimized theta values using gradient descent
feature_sample = lr.feature_normalize(np.array([1650, 3]))[0]
feature_sample = np.append(np.ones(1), feature_sample)
predict = lr.predict(feature_sample, theta)
