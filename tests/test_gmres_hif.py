# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

import numpy as np
from scipy.sparse import random
from scipy.sparse.linalg import norm
from hifir4py.ksp import gmres_hif


def test_random():
    while True:
        A = random(10, 10, 0.5)
        b = A * np.ones(10)
        if np.linalg.norm(b, ord=1) / norm(A, 1) >= 1e-10:
            break
    x, flag, _ = gmres_hif(A, b)
    assert flag == 0
    res = np.linalg.norm(x - 1) / np.linalg.norm(b)
    assert res <= 1e-6
