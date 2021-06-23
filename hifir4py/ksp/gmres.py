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
"""Right-preconditioned GMRES with HIF"""
import time
import typing
import numpy as np
import scipy.linalg as la
from . import mul_crs_ax
from ..hif import HIF
from ..utils import to_crs, must_1d, ensure_same, Tuple, List

__all__ = ["GMRES_WorkSpace", "gmres_hif", "gmres"]


class GMRES_WorkSpace:
    """Workspace class for GMRES solver

    This class is a collection of all buffer arrays used in GMRES solver, i.e.,
    :func:`.gmres_hif`. Users can create this buffer object *a priori*, so that
    it will be efficient to invoke GMRES multiple times.

    Attributes
    ----------
    y : :class:`~numpy.ndarray`
        Length of restart+1
    R : :class:`~numpy.ndarray`
        restart-restart upper triangular matrix used in Arnoldi
    Q : :class:`~numpy.ndarray`
        restart-n used in Arnoldi
    J : :class:`~numpy.ndarray`
        2-restart used in Given's rotation
    v, w : :class:`~numpy.ndarray`
        Length-n arrays
    w2 : :class:`~numpy.ndarray`
        Length of restart

    Examples
    --------
    The following code constructs a workspace buffer that is length 10 and with
    restart dimension 30.

    >>> from hifir4py.ksp import *
    >>> work = GMRES_WorkSpace(10)
    """

    def __init__(self, n: int, restart: int = 30, dtype: np.dtype = np.float64):
        """Constructor of workspace buffer for GMRES

        Parameters
        ----------
        n : int
            Length of the RHS vector
        restart : int, optional
            Restart dimension of KSP, default is 30
        dtype : :class:`~numpy.dtype`
            Data type, default is double-precision real
        """
        if n <= 0 or restart <= 0:
            raise ValueError("Invalid n ({}) or restart ({})".format(n, restart))
        self.y = np.zeros(restart + 1, dtype=dtype)
        self.R = np.zeros((restart, restart), dtype=dtype)
        self.Q = np.zeros((restart, n), dtype=dtype)
        self.J = np.zeros((2, restart), dtype=dtype)
        self.v = np.zeros(n, dtype=dtype)
        self.w = self.v.copy()
        self.w2 = np.zeros(restart, dtype=dtype)

    @property
    def dtype(self) -> np.dtype:
        """numpy.dtype: NumPy data type"""
        return self.y.dtype

    @property
    def size(self) -> int:
        """int: Size (length) of the workspace"""
        return self.v.size

    @property
    def restart(self) -> int:
        """int: Restart dimension"""
        return self.R.shape[0]


def gmres_hif(
    A, b, M: HIF = None, **kw
) -> Tuple[np.ndarray, int, dict]:  # pylint: disable=unsubscriptable-object
    r"""HIF-preconditioned (right) restarted GMRES

    This function implements the right-preconditioned GMRES(restart) with HIF
    preconditioners for consistent systems. This function can be either called
    as a "one-time" solver in that one can factorize HIF and compute the
    solution at the same time, or can be called repeatedly where the user can
    maintain the preconditioner and workspace buffer outside of this function.
    The implementation follows the standard right-preconditioned GMRES with
    restart; see *Iterative Methods for Sparse Linear Systems* by Saad.

    Parameters
    ----------
    A : :class:`~scipy.sparse.csr_matrix`
        Input CRS matrix
    b : :class:`~numpy.ndarray`
        Input RHS vector (1D)
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
    rtol : float, optional
        Relative tolerance threshold, default is 1e-6. The solver will terminate
        if :math:`\Vert b-Ax\Vert\le\text{rtol}\Vert b\Vert`.
    work : :class:`.GMRES_WorkSpace`, optional
        Workspace buffer, by default, the solver creates all buffer space.

    Returns
    -------
    x : :class:`~numpy.ndarray`
        The computed solution, which overwrites ``x0`` if passed in.
    flag : int
        Solver termination flags: The solver either successfully finishes (0),
        reaches the maximum iteration bound (1), breaks down due to NaN/Inf
        residuals (2), or stagnates (3).
    info : dict
        Statistics of the solver, which are stored in a dictionary with fields:
        total iterations (key ``"iters"``), final relative residual (key
        ``"relres"``), relative residual history array (key ``"resids"``), and
        factorization and solve runtimes (key ``"times"``).

    Notes
    -----
    For repeated calls of GMRES solver, one should at least maintain the buffer
    outside of this function, so that it won't waste time in memory allocation
    and deallocation. In addition, one can maintain the preconditioner object
    ``M`` and update it whenever necessary; this is useful in solving dynamic
    and/or transient problems.

    Notice that one can pass the key-value pairs ``**kw`` in
    :meth:`..HIF.factorize` to this function call if he/she plans to let this
    solver build preconditioner.

    Examples
    --------
    There are many ways one can use this GMRES solver.

    >>> import numpy as np
    >>> from scipy.sparse import rand
    >>> from hifir4py.ksp import *
    >>> A = rand(10, 10, 0.5)
    >>> b = A.dot(np.ones(10))

    The simplest way is

    >>> x, flag, _ = gmres_hif(A, b)

    which is equivalent to the following call

    >>> x, flag, _ = gmres_hif(A, b, M=None, verbose=1, restart=30, maxit=500,
    ... rtol=1e-6, x0=None, work=None)

    In addition, if ``M`` is ``None``, then one can pass any key-value pairs to
    this function that serve as control parameters of the preconditioner.

    The following illustrates how to maintain the preconditioner and workspace
    buffer outside of the solver.

    >>> from hifir4py import *
    >>> M = HIF(A)
    >>> work = GMRES_WorkSpace(b.size)
    >>> x, flag, info = gmres_hif(A, b, M=M, work=work)
    >>> # One can then update M and repeated call gmres_hif as above
    >>> print("total #iters = {}".format(info["iters"]))
    """
    must_1d(b)
    b = b.reshape(-1)
    A = to_crs(A)
    ensure_same(A.dtype, b.dtype, "Unmatched dtypes")
    n = b.size
    ensure_same(n, A.shape[0])
    verbose = kw.pop("verbose", 1)
    # compute factorization if necessary
    M, t_fac = _get_M(A, M, verbose, **kw)
    it = 0
    flag = 0
    x = kw.pop("x0", None)
    zero_start = False
    if x is None:
        zero_start = True
        x = np.zeros(n, dtype=b.dtype)
    must_1d(x)
    x = x.reshape(-1)
    ensure_same(x.size, n)
    beta0 = np.linalg.norm(b)
    if beta0 == 0.0:
        # quick return if possible
        x[:] = 0.0
        return (
            x,
            flag,
            {
                "iters": it,
                "relres": 0.0,
                "resids": np.asarray([], dtype=np.float64),
                "times": [t_fac, 0.0],
            },
        )
    # get matrix-vector kernel
    mul_ax_kernel = _select_mul_ax_kernel(np.iscomplexobj(A), A.indptr.dtype.itemsize * 8)
    restart, rtol, maxit = _determine_gmres_pars(30, 1e-6, 500, **kw)

    # Begin to time the solve part
    t_start = time.time()
    work = kw.pop("work", GMRES_WorkSpace(n, restart, b.dtype))
    if work is None:
        work = GMRES_WorkSpace(n, restart, b.dtype)
    else:
        if not issubclass(work.__class__, GMRES_WorkSpace):
            raise TypeError("Invalid workspace type")
    ensure_same(work.size, n)
    ensure_same(work.dtype, A.dtype, "Unmatched data types")
    max_outer_iters = int(np.ceil(maxit / restart))
    resids = np.empty(maxit)
    resid = 1.0

    # Main loop
    if verbose:
        print("Starting GMRES iterations...")
    for it_outer in range(max_outer_iters):
        if it_outer > 0 or not zero_start:
            mul_ax_kernel(A.indptr, A.indices, A.data, x, work.v)
            work.v[:] = b - work.v
            beta = np.linalg.norm(work.v)
        else:
            work.v[:] = b
            beta = beta0
        work.y[0] = beta
        work.Q[0, :] = work.v / beta
        j = 0
        while True:
            work.v[:] = work.Q[j, :]
            work.w = M.apply(work.v, x=work.w)
            mul_ax_kernel(A.indptr, A.indices, A.data, work.w, work.v)
            for k in range(j + 1):
                tmp = np.vdot(work.v, work.Q[k, :])
                work.v -= tmp * work.Q[k, :]
                work.w2[k] = tmp
            v_norm = np.linalg.norm(work.v)
            v_norm2 = v_norm * v_norm
            if j + 1 < restart:
                work.Q[j + 1, :] = work.v / v_norm
            # Given's rotation
            for col_j in range(j):
                tmp = work.w2[col_j]
                work.w2[col_j] = (
                    np.conj(work.J[0, col_j]) * tmp + np.conj(work.J[1, col_j]) * work.w2[col_j + 1]
                )
                work.w2[col_j + 1] = -work.J[1, col_j] * tmp + work.J[0, col_j] * work.w2[col_j + 1]
            rho = np.sqrt(np.conj(work.w2[j]) * work.w2[j] + v_norm2)
            work.J[0, j] = work.w2[j] / rho
            work.J[1, j] = v_norm / rho
            work.y[j + 1] = -work.J[1, j] * work.y[j]
            work.y[j] *= np.conj(work.J[0, j])
            work.w2[j] = rho
            work.R[: j + 1, j] = work.w2[: j + 1]
            resid_prev = resid
            resid = abs(work.y[j + 1]) / beta0
            if np.isnan(resid) or np.isinf(resid):
                flag = 2
                break
            if resid >= resid_prev * (1.0 - 1e-8):
                flag = 3
                break
            if it >= maxit:
                flag = 1
                break
            resids[it] = resid
            it += 1  # increment total iterations
            if resid <= rtol or j + 1 >= restart:
                break
            j += 1
        # finished inner inf loop
        # backsolve
        la.solve_triangular(
            work.R[: j + 1, : j + 1], work.y[: j + 1], lower=False, overwrite_b=True
        )
        np.dot(work.y[: j + 1], work.Q[: j + 1, :], out=work.v)
        work.w = M.apply(work.v, x=work.w)
        x += work.w
        if resid <= rtol or flag:
            break
    t_solve = time.time() - t_start
    if verbose:
        if flag == 0:
            print("Computed solution in {} iterations and {:.4g}s.".format(it, t_solve))
        elif flag == 2:
            print("GMRES broke down!")
        elif flag == 3:
            print("GMRES stagnated after {} iterations and {:.4g}s.".format(it, t_solve))
        else:
            print("GMRES failed to converge after {} iterations and {:.4g}s.".format(it, t_solve))
    return (
        x,
        flag,
        {
            "iters": it,
            "relres": resid,
            "resids": resids[:it],
            "times": [t_fac, t_solve],
        },
    )


def gmres(*args, **kw):
    """An alias of :func:`.gmres_hif`"""
    return gmres_hif(*args, **kw)


def _determine_gmres_pars(
    restart_default: int,
    rtol_default: typing.Union[float, List[float]],  # pylint: disable=unsubscriptable-object
    maxit_default: int,
    **kw
):
    """Helper function to determine core GMRES parameters"""
    restart = kw.pop("restart", restart_default)
    if restart is None or restart < 1:
        restart = restart_default
    rtol = kw.pop("rtol", rtol_default)
    if rtol is None or (np.isscalar(rtol) and rtol <= 0.0):
        rtol = rtol_default
    maxit = kw.pop("maxit", maxit_default)
    if maxit is None or maxit < 1:
        maxit = maxit_default
    return restart, rtol, maxit


def _select_mul_ax_kernel(is_complex: bool, index_size: int) -> typing.Callable[..., None]:
    """Helper function to select matrix-vector kernel"""
    if index_size not in (32, 64):
        raise ValueError("Must be either int32 or int64")
    v = "d" if not is_complex else "z"
    return getattr(mul_crs_ax, "{}i{}_multiply".format(v, index_size))


def _get_M(
    A, M: HIF, verbose: int, **kw
) -> Tuple[HIF, float]:  # pylint: disable=unsubscriptable-object
    """Helper function to get M"""
    if M is None:
        start = time.time()
        M = HIF(A, verbose=verbose > 1, **kw)
        t_fac = time.time() - start
        if verbose:
            print("HIF factorization finished in {:.4g}s.".format(t_fac))
    else:
        if not issubclass(M.__class__, HIF):
            raise TypeError("Must be (child of) HIF")
        if verbose:
            print("Preconditioned provided as input.")
        t_fac = 0.0
    return M, t_fac
