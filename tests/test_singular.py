# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HILUCSI4PY project                     #
###############################################################################

# matrix was created by MATLAB gallery('neumann', 100)
# This matrix has a constant mode in the null space

from scipy.io import loadmat
import numpy as np

from hilucsi4py import create_ksp


def test_singular():
    A, p_left = loadmat("singular.mat")["A"], loadmat("singular.mat")["p"].reshape(-1)
    ksp = create_ksp("fgmres")
    ksp.M.factorize(A)
    b = np.random.rand(A.shape[0])
    # filter out the left nullspace in b
    b -= (p_left.dot(b) / p_left.dot(p_left)) * p_left
    ksp.M.set_nsp_filter()  # enable constant mode filter
    x, _, _ = ksp.solve(A, b)
    res = np.linalg.norm(A * x - b) / np.linalg.norm(b)
    assert res <= 1e-6
