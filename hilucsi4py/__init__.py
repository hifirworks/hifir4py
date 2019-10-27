# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HILUCSI4PY project                     #
#                                                                             #
#    Copyright (C) 2019 NumGeom Group at Stony Brook University               #
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.   #
###############################################################################
from ._hilucsi4py import *

__version__ = version()


def get_include():
    """Get the include path

    Returns
    -------
    str
        Absolute path to this module
    """
    import os

    return os.path.dirname(os.path.abspath(__file__))


def create_ksp(ksp, *args, **kw):
    """Create and initialize a KSP solver with HILUCSI embedded in

    Parameters
    ----------
    ksp : str
        Solver name
    mixed : bool, optional
        Using mixed solver, default is False
    args, kw
        Positional and keyword arguments migrated to KSP constructor

    Returns
    -------
    KspSolver
        A KSP instance

    Raises
    ------
    ValueError
        Unknown KSP solver `ksp`

    Examples
    --------

    >>> from hilucsi4py import *
    >>> solver = create_ksp("gmres")  # GMRES
    >>> from scipy.sparse import random
    >>> A = random(10, 10, 0.5)
    >>> solver.M.factorize(A)
    >>> import numpy as np
    >>> b = np.random.rand(10)
    >>> x = solver.solve(A, b)

    Notes
    -----

    As to `ksp`, available options are

    #. ``gmres`` ([1]_, [2]_)
    #. ``qmrcgstab`` ([3]_)
    #. ``gmresr`` ([4]_)
    #. ``bicgstab`` ([5]_)

    special symbols ``show`` or ``query`` can be passed to get available KSP
    solver names, i.e.,

        >>> create_ksp("show")
        ["gmres", "qmrcgstab", "bicgstab", "gmresr"]

    References
    ----------

    .. [1]
        Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal
        residual algorithm for solving nonsymmetric linear systems. SIAM Journal
        on Scientific and Statistical Computing, 7(3), 856-869.
    .. [2]
        Saad, Y. (1993). A flexible inner-outer preconditioned GMRES
        algorithm. SIAM Journal on Scientific Computing, 14(2), 461-469.
    .. [3]
        Chan, T. F., Gallopoulos, E., Simoncini, V., Szeto, T., & Tong, C. H.
        (1994). A quasi-minimal residual variant of the Bi-CGSTAB algorithm for
        nonsymmetric systems. SIAM Journal on Scientific Computing, 15(2),
        338-347.
    .. [4]
        Van der Vorst, H. A., & Vuik, C. (1994). GMRESR: a family of nested
        GMRES methods. Numerical linear algebra with applications, 1(4), 369-386.
    .. [5]
        G. L. G. Sleijpen, D. R. Fokkema, and H. A van der Vorst. BiCGSTAB(J)
        and other hybrid Bi-CG methods. Numer. Algorithms, 7:75--109, 1994.
    """
    ksp = ksp.lower()
    if ksp in ("show", "query"):
        return ["gmres", "qmrcgstab", "bicgstab", "gmresr"]
    mixed = kw.pop("mixed", False)
    if ksp.find("gmresr") > -1:
        return TGMRESR(*args, **kw) if not mixed else TGMRESR_Mixed(*args, **kw)
    if ksp.find("gmres") > -1:
        return FGMRES(*args, **kw) if not mixed else FGMRES_Mixed(*args, *kw)
    if ksp.find("qmrcgstab") > -1:
        return FQMRCGSTAB(*args, **kw) if not mixed else FQMRCGSTAB_Mixed(*args, **kw)
    if ksp.find("bicgstab"):
        return FBICGSTAB(*args, **kw) if not mixed else FBICGSTAB_Mixed(*args, **kw)
    raise ValueError("Unknown KSP solver {}".format(ksp))


def create_M(mixed=False):
    """Create HILUCSI preconditioner

    Parameters
    ----------
    mixed : bool, optional
        If False (default), the do not enable mixed precision

    Returns
    -------
    HILUCSI or HILUCSI_Mixed
        If `mixed` is False, return the former.

    See Also
    --------
    create_ksp

    Notes
    -----

    This is designed for users who just need preconditioner. Otherwise, please
    directly use :func:`create_ksp` as each KSP solver has a preconditioner
    embedded in.
    """
    return HILUCSI() if not mixed else HILUCSI_Mixed()
