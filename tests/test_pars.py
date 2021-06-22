# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

from hifir4py import Params


def test_pars():
    params = Params()
    params["no_pre"] = 1
    assert params["no_pre"] == 1
    params.kappa = 1e10
    assert params["kappa_d"] == 1e10
    assert params["kappa"] == 1e10
