# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

from scipy.sparse import random
import numpy as np

from hifir4py import GMRES_Mixed


def test_mixed():
    A = random(10, 10, 0.5)
    solver = GMRES_Mixed()
    solver.M.factorize(A)
    b = A * np.ones(10)
    x, _ = solver.solve(A, b)
    res = np.linalg.norm(x - 1) / np.linalg.norm(b)
    assert res <= 1e-6
    assert abs(res - solver.resids[-1]) <= 1e-6
