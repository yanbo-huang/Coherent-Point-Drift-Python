from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import numpy as np
from cpd.cpd_p import cpd_p


def register_rigid(x, y, w, max_it=150):
    """
    Registers Y to X using the Coherent Point Drift algorithm, in rigid fashion.
    Note: For affine transformation, t = scale*y*r'+1*t'(* is dot). r is orthogonal rotation matrix here.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    max_it : int
        Maximum number of iterations. The default value is 150.

    Returns
    -------
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    # get dataset lengths and dimensions
    [n, d] = x.shape
    [m, d] = y.shape
    # t is the updated moving shape,we initialize it with y first.
    t = y
    # initialize sigma^2
    sigma2 = (m*np.trace(np.dot(np.transpose(x), x))+n*np.trace(np.dot(np.transpose(y), y)) -
              2*np.dot(sum(x), np.transpose(sum(y))))/(m*n*d)
    iter = 0
    while (iter < max_it) and (sigma2 > 10.e-8):
        [p1, pt1, px] = cpd_p(x, t, sigma2, w, m, n, d)
        # precompute
        Np = np.sum(pt1)
        mu_x = np.dot(np.transpose(x), pt1)/Np
        mu_y = np.dot(np.transpose(y), p1)/Np
        # solve for Rotation, scaling, translation and sigma^2
        a = np.dot(np.transpose(px), y)-Np*(np.dot(mu_x, np.transpose(mu_y)))
        [u, s, v] = np.linalg.svd(a)
        s = np.diag(s)
        c = np.eye(d)
        c[-1, -1] = np.linalg.det(np.dot(u, v))
        r = np.dot(u, np.dot(c, v))
        scale = np.trace(np.dot(s, c))/(sum(sum(y*y*np.matlib.repmat(p1, 1, d)))-Np *
                                        np.dot(np.transpose(mu_y), mu_y))
        sigma22 = np.abs(sum(sum(x*x*np.matlib.repmat(pt1, 1, d)))-Np *
                         np.dot(np.transpose(mu_x), mu_x)-scale*np.trace(np.dot(s, c)))/(Np*d)
        sigma2 = sigma22[0][0]
        # ts is translation
        ts = mu_x-np.dot(scale*r, mu_y)
        t = np.dot(scale*y, np.transpose(r))+np.matlib.repmat(np.transpose(ts), m, 1)
        iter = iter+1
    return t
