# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# Scientific and vector computation for python
import numpy as np

import regression as lr


def plot_data(feature, label, xmin, xmax, ymin, ymax, xlabel, ylabel):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    feature : array_like
        Data point values for x-axis.

    label : array_like
        Data point values for y-axis. Note x and y should have the same size.

    xmin, xmax, ymin, ymax : float, optional
                             The axis limits to be set. Either none or all of the limits must
                             be given. This can also be achieved using ::

                                 ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    xlabel : string
        x-axis label.

    ylabel : string
        y-axis label.

    """
    fig = pyplot.figure()  # open a new figure

    pyplot.plot(feature, label, 'ro', ms=10)

    pyplot.ylabel(ylabel)

    pyplot.xlabel(xlabel)

    pyplot.axis([xmin, xmax, ymin, ymax])

    pass


def plot_check_cost(cost_history):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    cost_history : array_like
        cost function J history.

    """

    # Generating values for theta0, theta1 and the resulting cost value

    pyplot.plot(cost_history)
    pyplot.xlabel('Iteration')
    pyplot.ylabel('J_Theta')
    pyplot.title('Cost function using Gradient Descent')

    pass


def plot_cost(feature, label, theta):
    """
    Plots to show 𝐽(𝜃)  varies with changes in  𝜃0  and  𝜃1 . The cost function  𝐽(𝜃)
    is bowl-shaped and has a global minimum.

    Parameters
    ----------
    feature : array_like
        Data point values for x-axis.

    label : array_like
        Data point values for y-axis. Note x and y should have the same size.

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    """

    # grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    cost_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            cost_vals[i, j] = lr.compute_cost(feature, label, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped

    # surface plot
    fig = pyplot.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(theta0_vals, theta1_vals, cost_vals, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    pyplot.title('Surface')

    # rotate for better angle
    ax.view_init(30, 120)
    pass
