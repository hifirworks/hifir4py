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
=================================================
HIFIR Preconditioner for Python (:mod:`hifir4py`)
=================================================

.. currentmodule:: hifir4py

Contents
========

The HIF preconditioner
----------------------

.. autosummary::
    :template: hifir4py_autosummary.rst
    :toctree: generated/

    HIF - Main object for HIF preconditioner


Control Parameters
------------------

.. autosummary::
    :template: hifir4py_autosummary.rst
    :toctree: generated/

    Params - Dictionary-like object for control parameters
    Verbose - Verbose levels
    Reorder - Reordering strategies
    Pivoting - Pivoting options

Example Usage
=============

There are two aspects in using this software package:

1. Compute a HIF preconditioner, and
2. apply a HIF preconditioner (with iterative refinements).

In addition, we demonstrate how to use HIF directly through KSP solvers.

Factorization
-------------

>>> from hifir4py import *
>>> from scipy.sparse import rand
>>> import numpy as np

Now, to factorize a matrix, one can simply do

>>> A = rand(10, 10, 0.5)
>>> M = HIF(A)

or, alternatively, perform

>>> M = HIF()
>>> M.factorize(A)

In many applications, the preconditioner may be built based on a "sparser"
system (sparsifier), such as p-multigrid. One can provide such sparsifier
in factorization step.

>>> S = rand(10, 10, 0.3)
>>> M = HIF(A, S=S)

or, alternatively, perform

>>> M = HIF()
>>> M.factorize(A, S=S)

The default parameters may be too conservative, and one may want to optimize
parameters for his/her applications.

>>> params = Params()
>>> params
{'tau_L': 0.0001, 'tau_U': 0.0001, 'kappa_d': 3.0, 'kappa': 3.0,
'alpha_L': 10.0, 'alpha_U': 10.0, 'rho': 0.5, 'c_d': 10.0, 'c_h': 2.0, 'N': -1,
'verbose': 1, 'rf_par': 1, 'reorder': 0, 'spd': 0, 'check': 1, 'pre_scale': 0,
'symm_pre_lvls': 1, 'threads': 0, 'mumps_blr': 1, 'fat_schur_1st': 0,
'rrqr_cond': 0.0, 'pivot': 2, 'gamma': 1.0, 'beta': 1000.0, 'is_symm': 0,
'no_pre': 0}
>>> M = HIF(A, params=params)

To disable verbose, one can do

>>> params["verbose"] = Verbose.NONE
>>> params.disable_verbose()  # equivalent to above

To set larger drop tolerances (:math:`\tau`) and conditioning thresholds
(:math:`\kappa`) and smaller scalability-oriented dropping factors
(:math:`\alpha`), one can set the ``"tau_L"`` and ``"tau_U"`` entries,
``"kappa_d"`` and ``"kappa"`` entries, and ``"alpha_L"`` and ``"alpha_U"``
entries.

>>> params["tau_L"] = params["tau_L"] = 1e-2
>>> params["kappa_d"] = params["kappa"] = 5
>>> params["alpha_L"] = params["alpha_U"] = 3

or, equivalently, using the following methods to set uniform values.

>>> params.tau = 1e-2
>>> params.kappa = 5
>>> params.alpha = 3

Applying HIF
------------

There are four modes for applying HIF preconditioners:

1. multilevel triangular solve (most commonly used),
2. transpose/Hermitian multilevel triangular solve,
3. multilevel matrix-vector multiplication, and
4. transpose/Hermitian multilevel matrix-vector multiplication.

Iterative refinements can be enabled in triangular solve modes (modes 1 and 2),
and all four operations can be used in a unified function :meth:`~.HIF.apply`.

>>> b = np.random.rand(10)

The following code illustrates how to perform the standard triangular solve,
i.e., :math:`\boldsymbol{x}=\boldsymbol{M}^g\boldsymbol{b}`.

>>> x = M.apply(b)
>>> x = M.apply(b, op="S")  # equivalent to above

Iterative refinements can be enabled through

>>> x = M.apply(b, nirs=2)  # two-step IR

Similarly, matrix-vector multiplication, i.e.,
:math:`\boldsymbol{x}=\boldsymbol{M}\boldsymbol{b}`, can be done through the
following code.

>>> x = M.apply(b, op="M")

Finally, ``op="SH"`` and ``op="MH"`` enable the tranpose/Hermitian option for
multilevel triangular solve and matrix-vector multiplication, respectively.

Using in KSP
------------

``hifir4py`` has built-in support for using in SciPy's KSP solvers.

>>> from scipy.sparse.linalg import gmres
>>> x, flag = gmres(A, b, M=M.to_scipy())

.. note:: SciPy uses left-preconditioned GMRES, which is not recommended.
"""
from ._hifir import version
from .hif import *  # noqa: F401, F403
from . import ksp  # noqa: F401

__version__ = version()
