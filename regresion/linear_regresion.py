# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np
from numpy.linalg import inv


def compute_cost(feature, label, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in feature (X) and label (y).

    Parameters
    ----------
    feature : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.

    label : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    cost : float
        The value of the regression cost function.
    """

    # initialize some useful values
    total_examples = len(label)  # number of training examples (m)

    predictions = feature.dot(theta)  # Predictions of the hypothesis on all samples

    sqr_errors = (predictions - label) ** 2

    cost = 1 / (2 * total_examples) * np.sum(sqr_errors)

    return cost


def gradient_descent(feature, label, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    feature : array_like
        The input dataset of shape (m x n+1).

    label : array_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    cost_history : list
        A python list for the values of the cost function after each iteration.

    """
    # Initialize some useful values
    total_examples = len(label)  # number of training examples

    cost_history = []  # Use a python list to save cost in every iteration

    for i in range(num_iters):
        predictions = feature.dot(theta)

        error = np.dot(feature.transpose(), (predictions - label))

        descent = alpha * 1 / total_examples * error

        theta -= descent

        # save the cost in every iteration
        cost_history.append(compute_cost(feature, label, theta))

    return theta, cost_history


def feature_normalize(feature):
    """
    Normalizes the features in feature. returns a normalized version of feature where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    feature : array_like
        The dataset of shape (m x n).

    Returns
    -------
    feature_norm : array_like
        The normalized dataset of shape (m x n).

    """

    mean = np.mean(feature, axis=0)

    std = np.std(feature, axis=0)

    feature_norm = (feature - mean) / std

    return feature_norm, mean, std


def normal_eqn(feature, label):
    """
    Computes the closed-form solution to linear regression using the normal equations.

    Parameters
    ----------
    feature : array_like
        The dataset of shape (m x n+1).

    label : array_like
        The value at each data point. A vector of shape (m, ).

    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).

    """
    theta = np.zeros(feature.shape[1])

    feature_transpose = feature.transpose()  # transpose feature array

    theta = inv(feature_transpose.dot(feature)).dot(feature_transpose).dot(label)

    return theta


def predict(to_predict, theta):
    """
    Takes in numpy array of to_predict and theta and return the predicted value of y based on theta
    """

    predictions = np.dot(theta.transpose(), to_predict)

    return predictions[0]
