# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

import numpy as np
from scipy.sparse import random
from hifir4py import HIF


def test_multiply():
    A = random(30, 30, 0.5)
    M = HIF()
    M.factorize(A)
    b = A * np.ones(30)
    x1 = M.apply(b)
    x2 = M.apply(x1, op="M")
    assert np.linalg.norm(x2 - b) / np.linalg.norm(b) <= 1e-10


def test_multiply_trans():
    A = random(30, 30, 0.5)
    M = HIF()
    M.factorize(A, is_mixed=True)
    b = A * np.ones(30)
    x1 = M.apply(b, op="SH")
    x2 = M.apply(x1, op="MH")
    assert np.linalg.norm(x2 - b) / np.linalg.norm(b) <= 1e-5
