# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

from hifir4py import GMRES, Params


def test_ksp_pars():
    solver = GMRES()
    print(solver)
    assert solver.rtol == 1e-6
    solver.maxit = 100
    assert solver.maxit == 100


def test_pars():
    params = Params()
    params["no_pre"] = 1
    assert params["no_pre"] == 1
