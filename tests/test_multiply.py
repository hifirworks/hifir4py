# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

from scipy.sparse import random
import numpy as np

from hifir4py import HIF


def test_multiply():
    A = random(30, 30, 0.5)
    M = HIF()
    M.factorize(A)
    b = A * np.ones(30)
    x1 = M.solve(b)
    x2 = M.mmultiply(x1)
    assert np.linalg.norm(x2 - b) / np.linalg.norm(b) <= 1e-12


def test_multiply_trans():
    A = random(30, 30, 0.5)
    M = HIF()
    M.factorize(A)
    b = A * np.ones(30)
    x1 = M.solve(b, trans=True)
    x2 = M.mmultiply(x1, trans=True)
    assert np.linalg.norm(x2 - b) / np.linalg.norm(b) <= 1e-12
