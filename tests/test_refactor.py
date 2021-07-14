# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

from scipy.sparse import rand
from hifir4py import HIF


def test_random():
    M = HIF(rand(10, 10, 0.5))
    A = M.A
    M.refactorize(rand(10, 10, 0.5))
    assert A is M.A
