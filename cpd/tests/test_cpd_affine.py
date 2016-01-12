from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import numpy as np
import numpy.testing as npt
import cpd
import os


def _load_ndarray(file_name):
	base_dir = os.path.dirname(os.path.abspath(__file__))
	file_path = os.path.join(os.path.join(base_dir, "data"), file_name)
	return np.load(file_path)


def test_register_affine():
	# load input dataset
	X = _load_ndarray("affine_X.npy")
	Y = _load_ndarray("affine_Y.npy")
	# load expected output
	T_desired = _load_ndarray("affine_T.npy")
	# call function under test
	T_actual = cpd.register_affine(X, Y, w=0.6)
	# compare outputs
	npt.assert_almost_equal(T_actual, T_desired)
