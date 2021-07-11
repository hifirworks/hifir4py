# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
#                                                                             #
#   Copyright (C) 2019--2021 NumGeom Group at Stony Brook University          #
#                                                                             #
#   This program is free software: you can redistribute it and/or modify      #
#   it under the terms of the GNU Affero General Public License as published  #
#   by the Free Software Foundation, either version 3 of the License, or      #
#   (at your option) any later version.                                       #
#                                                                             #
#   This program is distributed in the hope that it will be useful,           #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of            #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
#   GNU Affero General Public License for more details.                       #
#                                                                             #
#   You should have received a copy of the GNU Affero General Public License  #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.    #
###############################################################################
r"""
=============================================
KSP Solvers Using HIFIR (:mod:`hifir4py.ksp`)
=============================================

.. currentmodule:: hifir4py.ksp

Contents
========

Solvers
-------

.. autosummary::
    :toctree: generated/

    gmres_hif - HIF-preconditioned restarted GMRES
    gmres - Alias of :func:`.gmres_hif`
    pipit_hifir - HIFIR-based pseudoinverse solution solver
    pipit - Alias of :func:`.pipit_hifir`
    orth_null - Solver for computing multiple null-space vectors

.. note::

    :func:`.gmres_hif` is designed for solving consistent systems, in that the
    right-hand side is in :math:`\mathcal{R}(\boldsymbol{A})`. On the other
    hand, :func:`.pipit_hifir` is designed for seeking the pseudoinverse
    solution of (potentially) inconsistent systems; it is based on a combination
    of :func:`.orth_null` and :func:`.gmres_hif`. The users should not call
    :func:`.orth_null` in general.

Miscellany
----------

.. autosummary::
    :template: hifir4py_autosummary.rst
    :toctree: generated/

    GMRES_WorkSpace - Work space buffer for :func:`.gmres_hif`
    FGMRES_WorkSpace - Work space buffer for :func:`.pipit_hifir` and :func:`.orth_null`

.. note::

    :class:`.FGMRES_WorkSpace` can also be used in :func:`.gmres_hif`.
    Both :class:`.GMRES_WorkSpace` and :class:`.FGMRES_WorkSpace` are optional,
    but should be used for getting good performance for sequences of calls of
    solvers.

Example Usage
=============

Here, we only show the simplest and most straightforward way to use our
HIF-preconditioned GMRES and PIPIT solvers. For advanced usage, refer to
the examples in :func:`.gmres_hif` and :func:`.pipit_hifir`.

HIF-preconditioned GMRES(:math:`m`)
------------------------------------

>>> import numpy as np
>>> from scipy.sparse import rand
>>> from hifir4py import *
>>> A = rand(20, 20, 0.5)
>>> b = A.dot(np.ones(20))
>>> x, flag, info = gmres_hif(A, b)  # HIF is factorized inside gmres_hif
>>> assert flag == 0, "GMRES failed with flag={}".format(flag)
>>> print("GMRES iterations are {}".format(info["iters"]))

We refer the readers to this more comprehensive :ref:`example <demo_gmres>`.

PIPIT solver
------------

We refer the readers to this more comprehensive :ref:`example <demo_pipit>`,
which solves a linear elasticity with pure traction boundary conditions in 3D.
This system has six-dimensional null space and is inconsistent.

"""
from .gmres import GMRES_WorkSpace, gmres_hif, gmres  # noqa: F401
from .orth_null import FGMRES_WorkSpace, orth_null  # noqa: F401
from .pipit import pipit_hifir, pipit  # noqa: F401
