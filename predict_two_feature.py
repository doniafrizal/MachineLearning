"""
Prediction one feature

"""

# used for manipulating directory path
import os
import pandas as pd

# Scientific and vector computation for python
import numpy as np

# Import local module
import regression as lr

# Read comma separated data

data = pd.read_csv(os.path.join('data',
                                'ex1data2.txt'), delimiter=',', header=None)

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
ITERATIONS = 1400
ALPHA = 0.1

theta, cost_history = lr.gradient_descent(feature_norm,
                                          label, theta, ALPHA, ITERATIONS)
print('Theta found by gradient descent: ' + str(round(theta[0, 0], 2)) +
      ' , ' + str(round(theta[1, 0], 2)) + ' , '
      + str(round(theta[2, 0], 2)))

# Plot cost function to verify minimum cost achieved.
lr.plot_check_cost(cost_history)

# Predict values using optimized theta values using gradient descent
feature_sample = lr.feature_normalize(np.array([1650, 3]))[0]
feature_sample = np.append(np.ones(1), feature_sample)
predict = lr.predict(feature_sample, theta)
