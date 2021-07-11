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
"""PIPIT solver for pseudoinverse (aka minimal-norm) solution"""
import time
import numpy as np
import scipy.sparse.linalg as spla
from .orth_null import FGMRES_WorkSpace, orth_null
from .gmres import gmres_hif, _get_M, _determine_gmres_pars
from ..utils import to_crs, must_1d, ensure_same, Tuple, Dict

__all__ = ["pipit_hifir", "pipit"]


def pipit_hifir(  # noqa: C901
    A, b: np.ndarray, n_null: int, M=None, **kw
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, dict]:  # pylint: disable=unsubscriptable-object
    """PIPIT solver for solving the pseudoinverse solution

    PIPIT implements a pseudoinverse solution solver with small dimension of
    nullspace. The system can be potentially inconsistent in that the RHS
    vector may not fully in the range of A.

    Parameters
    ----------
    A : :class:`~scipy.sparse.csr_matrix`
        User input matrix
    b : :class:`~numpy.ndarray`
        User input RHS
    n_null : int
        Nullspace dimension (>0)
    M : :class:`~hifir4py.HIF`, optional
        HIF preconditioner
    x0 : :class:`~numpy.ndarray`, optional
        Initial guess, default is zeros; if passed in, this will be overwritten
        by the solution.
    verbose : {0,1,2}, optional
        Verbose level, 0 disables the verbose printing, while 1 enables HIF
        verbose logging; default is 1.
    restart : int, optional
        Restart dimension, default is 30
    rtols : list, optional
        Legnth-two list storing the relative tolerance thresholds, default is
        [1e-6, 1e-12], where the former is for solving least-squares solution
        using HIF-GMRES, and the latter is for solving nullspaces.
    work : :class:`.FGMRES_WorkSpace`, optional
        Workspace buffer, by default, the solver creates all buffer space.
    vs : :class:`~numpy.ndarray`, optional
        User-provided right nullspace; if not given, then ``vs`` will be
        computed on-the-fly.

    Returns
    -------
    x : :class:`~numpy.ndarray`
        The computed pseudoinverse solution, which overwrites ``x0`` if provided
    ns : dict
        A dictionary of ``{"us": lns, "vs": rns}`` where ``lns`` and ``rns``
        are left and right null-space components, respectively.
    flag : int
        Termination flag for LS solver.
    info : dict
        Statistics for LS solver plus runtimes for factorization, left nullspace
        computation, LS computation, and right nullspace computation stages,
        i.e., ``info["times"]=[t_fac,t_lns,t_ls,t_rns]``.

    See Also
    --------
    :func:`~hifir4py.ksp.gmres_hif`
    """
    assert n_null > 0
    must_1d(b)
    b = b.reshape(-1)
    A = to_crs(A)
    ensure_same(A.dtype, b.dtype, "Unmatched dtypes")
    n = b.size
    ensure_same(n, A.shape[0])
    verbose = kw.pop("verbose", 1)
    # compute factorization if necessary
    M, t_fac = _get_M(A, M, verbose, **kw)
    restart, _, maxit = _determine_gmres_pars(30, None, 500, **kw)
    rtols = kw.pop("rtols", [1e-6, 1e-12])
    if rtols is None:
        rtols = [1e-6, 1e-12]
    if np.isscalar(rtols):
        rtol_null = 1e-12
        rtol = rtols
    else:
        rtol, rtol_null = rtols[0], rtols[1]
        if rtol_null <= 0.0:
            rtol_null = 1e-12
    if rtol <= 0.0:
        rtol = 1e-6

    work = kw.pop("work", FGMRES_WorkSpace(n, restart, b.dtype))
    if work is None:
        work = FGMRES_WorkSpace(n, restart, b.dtype)
    if not issubclass(work.__class__, FGMRES_WorkSpace):
        raise TypeError("Invalid workspace type")

    # Compute left nullspace
    t_leftnull = time.time()
    if verbose:
        print("\nStarting computing left nullspace...")
    us, its_left, its_ir_left = orth_null(
        A, n_null, M, True, restart, rtol_null, maxit, work=work, verbose=verbose
    )
    # projecting off the null space
    if us.size:
        b = b.copy()
        n_null = us.shape[1]
        for i in range(n_null):
            b -= np.vdot(us[:, i], b) * us[:, i]
    t_leftnull = time.time() - t_leftnull
    if verbose:
        print(
            "Finished left nullspace computation with total {} GMRES iterations".format(
                sum(its_left)
            )
        )
        print("and total {} inner refinements in {:.4g}s.".format(sum(its_ir_left), t_leftnull))

    if verbose:
        print("\nStarting GMRES for least-squares solution...")
    x, flag, stats = gmres_hif(
        A, b, M=M, restart=restart, rtol=rtol, maxit=maxit, work=work, verbose=verbose, **kw
    )
    t_ls = stats["times"][1]
    stats["times"] = [t_fac, t_leftnull, t_ls, 0.0]

    # PI solution if possible
    vs = kw.pop("vs", None)
    if vs is not None:
        ensure_same(vs.shape[0], us.shape[0])
        if verbose:
            print("\nUsing user-provided right nullspace")
        null_right = 1 if len(vs.shape) == 1 else vs.shape[1]
        for i in range(null_right):
            x -= np.vdot(vs[:, i], x) * vs[:, i]
        return x, {"us": us, "vs": vs}, flag, stats
    if spla.norm(A - A.H, ord=1) <= rtol_null * spla.norm(A, ord=1):
        if verbose:
            print("\nSystem is numerically symmetric; let vs=us.")
        vs = us
    else:
        # Compute right nullspace
        t_rightnull = time.time()
        if verbose:
            print("\nStarting computing right nullspace...")
        vs, its_right, its_ir_right = orth_null(
            A, n_null, M, False, restart, rtol_null, maxit, work=work, verbose=verbose
        )
        t_rightnull = time.time() - t_rightnull
        if verbose:
            print(
                "Finished right nullspace computation with total {} GMRES iterations".format(
                    sum(its_right)
                )
            )
            print(
                "and total {} inner refinements in {:.4g}s.".format(sum(its_ir_right), t_rightnull)
            )
        stats["times"][3] = t_rightnull
    # projecting off the null space
    n_null = vs.shape[1]
    for i in range(n_null):
        x -= np.vdot(vs[:, i], x) * vs[:, i]
    return x, {"us": us, "vs": vs}, flag, stats


def pipit(*args, **kw):
    """Alias of :func:`.pipit_hifir`"""
    return pipit_hifir(*args, **kw)
