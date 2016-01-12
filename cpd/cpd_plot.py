from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cpd_plot(x, y, t):
    """
    Plot the initial datasets and registration results.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    if len(x[0, :]) == 2:
        plt.figure(1)
        plt.plot(x[:, 0], x[:, 1], 'go')
        plt.plot(y[:, 0], y[:, 1], 'r+')
        plt.title("Before registration")
        plt.figure(2)
        plt.plot(x[:, 0], x[:, 1], 'go')
        plt.plot(t[:, 0], t[:, 1], 'r+')
        plt.title("After registration")
        plt.show()
    elif len(x[0, :]) == 3:
        ax1 = Axes3D(plt.figure(1))
        ax1.plot(x[:, 0], x[:, 1], x[:, 2], 'yo')
        ax1.plot(y[:, 0], y[:, 1], y[:, 2], 'r+')
        ax1.set_title("Before registration", fontdict=None, loc='center')
        ax2 = Axes3D(plt.figure(2))
        ax2.plot(x[:, 0], x[:, 1], x[:, 2], 'yo')
        ax2.plot(t[:, 0], t[:, 1], t[:, 2], 'r+')
        ax2.set_title("After registration", fontdict=None, loc='center')
        plt.show()
