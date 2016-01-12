from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import numpy as np


def cpd_r(n):
    """
    Calculating a random orthogonal 2d or 3d rotation matrix which satisfies det(r)=1.

    Parameters
    ----------
    n : int
        Rotation matrix's dimension.

    Returns
    -------
    r : ndarray
        Rotation matrix.
    """

    if n == 3:
        r1 = np.eye(3)
        r2 = np.eye(3)
        r3 = np.eye(3)
        r1[0: 2:, 0: 2] = rot(np.random.rand(1)[0])
        r2[:: 2, :: 2] = rot(np.random.rand(1)[0])
        r3[1:, 1:] = rot(np.random.rand(1)[0])
        r = np.dot(np.dot(r1, r2), r3)
    elif n == 2:
        r = rot(np.random.rand(1)[0])
    return r


def rot(f):
    """
    Generating a 2d random orthogonal rotation matrix.

    Parameters
    ----------
    f : float
        Random float number. Value is expected to be in range [0.0, 1.0].

    Returns
    -------
    r : ndarray
        2d random orthogonal rotation matrix.
    """
    r = np.array([[np.cos(f), -np.sin(f)],[np.sin(f), np.cos(f)]])
    return r



def cpd_b(n):
    """
    Generating a random 2d or 3d rotaiton matrix. Note: the rotation matrix don't need to satisfy det(b)=1.

    Parameters
    ----------
    n : int
        Rotation matrix's dimension.

    Returns
    -------
    b : ndarray
        Random rotation matrix.
    """   
    b = cpd_r(n)+abs(0.1*np.random.randn(n, n))
    return b

