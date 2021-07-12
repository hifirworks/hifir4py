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
"""Right-preconditioned FGMRES with HIFIR for solving nullspace"""
import typing
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from .gmres import GMRES_WorkSpace, _select_mul_ax_kernel
from ..utils import to_crs, Tuple

__all__ = ["FGMRES_WorkSpace", "orth_null"]


class FGMRES_WorkSpace(GMRES_WorkSpace):
    """Workspace buffer for flexible GMRES (FGMRES)

    Attributes
    ----------
    Z : :class:`~numpy.ndarray`
        restart-n used in storing preconditioned Q
    u, u2 : :class:`~numpy.ndarray`
        size-n array
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.Z = self.Q.copy()
        self.u = self.v.copy()
        self.u2 = self.v.copy()


def orth_null(
    A,  # CRS
    n_null: int,
    M,  # HIF
    leftnull: bool,
    restart: int,
    rtol: float,
    maxit: int,
    work: typing.Optional[FGMRES_WorkSpace] = None,
    verbose: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # pylint: disable=unsubscriptable-object
    """Compute multiple (small dimension) null-space components

    Parameters
    ----------
    A : :class:`~scipy.sparse.csr_matrix`
        Input coefficient matrix
    n_null : int
        Dimension of nullspace
    M : :class:`~hifir4py.HIF`
        HIF preconditioner (pre-factorized)
    leftnull : bool
        Whether or not computing for left nullspace
    restart : int
        FGMRES restart dimension
    rtol : float
        Relative nullspace residual tolerance
    maxit : int
        Maximum iterations
    work : :class:`.FGMRES_WorkSpace`, optional
        Buffer space
    verbose : int, optional
        Verbose level, default is 1; if verbose>1, then will print information
        for each nullspace vector computation

    Returns
    -------
    us : :class:`~numpy.ndarray`
        Null-space components (normalized), stored in Fortran order and
        shape[1] is at most n_null
    its : :class:`~numpy.ndarray`
        History of number of FGMRES iteration for all null-space components
    its_ir : :class:`~numpy.ndarray`
        History of number of iterative refinements for all null-space components
    """
    A = to_crs(A)
    mul_ax_func = _select_mul_ax_kernel(np.iscomplexobj(A), A.indptr.dtype.itemsize * 8)
    n = A.shape[0]
    if work is None:
        work = FGMRES_WorkSpace(n, restart, A.data.dtype)
    else:
        if not issubclass(work.__class__, FGMRES_WorkSpace):
            raise TypeError("workspace must be FGMRES_WorkSpace")
    bs = _generate_rand_orth(n, n_null)
    null_it = 0
    its = np.empty(n_null, dtype=int)
    its_ir = its.copy()
    x = np.zeros(n, dtype=work.dtype)
    op = "S" if not leftnull else "SH"
    if verbose > 1:
        print("Starting computing {} null-space components...".format(null_it))
    while True:
        if M.schur_size == M.schur_rank:
            # Preconditioner is not singular, apply IR to the RHS
            work.v[:] = bs[:, null_it]
            bs[:, null_it], _, __ = M.apply(
                work.v, op=op, x=bs[:, null_it], nirs=16, betas=[0.2, 1e8]
            )
            bs[:, null_it] /= np.linalg.norm(bs[:, null_it])
        x, flag, its_k, its_ir_k = _fgmres_null(
            A,
            bs[:, null_it],
            M,
            leftnull,
            restart,
            rtol,
            maxit,
            x,
            True,
            work,
            mul_ax_func,
        )
        if verbose > 1:
            if flag == 0:
                print(
                    "Finished the {} component with {} GMRES iterations and".format(null_it, its_k)
                )
                print("{} inner refinements.".format(its_ir_k))
            elif flag == 1:
                print("Reached maximum iterations for {} component.".format(null_it))
            elif flag == 3:
                print(
                    "{} null-space component stagnated with {} GMRES iterations".format(
                        null_it, its_k
                    )
                )
                print("and {} inner refinements.".format(its_ir_k))
        if flag:
            break
        bs[:, null_it] = x
        its[null_it] = its_k
        its_ir[null_it] = its_ir_k
        null_it += 1
        if null_it >= n_null:
            break
        x[:] = 0.0
    us = _make_orth(bs[:, :null_it])
    return us, its[:null_it], its_ir[:null_it]


def _generate_rand_orth(n: int, m: int) -> np.ndarray:
    """Generate random orthogonal RHS"""
    bs, _ = np.linalg.qr(np.random.rand(n, m))
    return np.asfortranarray(bs)


def _make_orth(us: np.ndarray) -> np.ndarray:
    if us.size == 0:
        return us
    us, _ = np.linalg.qr(us)
    return np.asfortranarray(us)


def _fgmres_null(
    A,  # CRS
    b: np.ndarray,
    M,  # HIF
    leftnull: bool,
    restart: int,
    rtol: float,
    maxit: int,
    x: np.ndarray,
    zero_start: bool,
    work: FGMRES_WorkSpace,
    mul_ax_func: typing.Callable[..., None],
) -> Tuple[np.ndarray, int, int, int]:  # pylint: disable=unsubscriptable-object
    """Single nullspace solver using customized HIFIR-FGMRES with Householder"""
    max_outer_iters = int(np.ceil(maxit / restart))
    flag = 0
    it = 0
    ref_iters = 0
    beta0 = np.linalg.norm(b)
    if beta0 == 0.0:
        x[:] = 0.0
        return x, flag, it, ref_iters
    null_res_prev = np.finfo(np.float64).max
    # compute 1-norm
    A_1nrm = spla.norm(A, ord=1)
    inrm_buf = np.empty(restart + 1)
    n = b.size
    form_x_thres = min(np.sqrt(1.0 / rtol), 1e6)
    op = "S" if not leftnull else "SH"

    # Main loop begins
    for it_outer in range(max_outer_iters):
        if it_outer > 0 or not zero_start:
            mul_ax_func(A.indptr, A.indices, A.data, x, work.w, trans=leftnull)
            work.u[:] = b - work.w
            beta = np.linalg.norm(work.u)
        else:
            work.u[:] = b
            beta = beta0
        beta2 = beta * beta

        # First Householder vector
        if work.u[0] < 0.0:
            beta = -beta
        updated_norm = np.sqrt(2.0 * beta2 + 2.0 * np.real(work.u[0]) * beta)
        work.u[0] += beta
        work.u /= updated_norm

        # The first Householder entry
        work.y[0] = -beta
        work.Q[0, :] = work.u

        # Inner iterations
        j = 0
        nirs = 1 << (it_outer + 3)
        inrm = 0.0
        nrm = 0.0
        while True:
            work.v[:] = -2.0 * np.conj(work.Q[j, j]) * work.Q[j, :]
            work.v[j] += 1.0
            for i in range(j - 1, -1, -1):
                work.v[i:] -= 2.0 * np.vdot(work.Q[i, i:], work.v[i:]) * work.Q[i, i:]
            work.v /= np.linalg.norm(work.v)
            work.u2[:] = work.v
            if it == 0:
                nirs = min(nirs, 4)
            # IR
            work.v, ref_its, _ = M.apply(work.u2, op=op, x=work.v, nirs=nirs, betas=[0.2, 10.0])
            ref_iters += ref_its
            work.Z[j, :] = work.v
            mul_ax_func(A.indptr, A.indices, A.data, work.v, work.w, trans=leftnull)

            # Orthogonalize the Krylov vector
            for i in range(j + 1):
                work.w[i:] -= 2.0 * np.vdot(work.Q[i, i:], work.w[i:]) * work.Q[i, i:]
            # Update the rotators
            if j < n:
                work.u[j] = 0.0
                work.u[j + 1] = work.w[j + 1]
                alpha2 = np.conj(work.w[j + 1]) * work.w[j + 1]
                work.u[j + 2 :] = work.w[j + 2 :]
                alpha2 += np.vdot(work.w[j + 2 :], work.w[j + 2 :])
                if alpha2 > 0.0:
                    alpha = np.sqrt(alpha2)
                    if work.u[j + 1] < 0.0:
                        alpha = -alpha
                    if j + 1 < restart:
                        updated_norm = np.sqrt(2.0 * alpha2 + 2.0 * np.real(work.u[j + 1]) * alpha)
                        work.u[j + 1] += alpha
                        work.Q[j + 1, j + 1 :] = work.u[j + 1 :] / updated_norm
                    work.w[j + 2 :] = 0.0
                    work.w[j + 1] = -alpha

            # Apply Given's rotation
            # Given's rotation
            for col_j in range(j):
                tmp = work.w[col_j]
                work.w[col_j] = (
                    np.conj(work.J[0, col_j]) * tmp + np.conj(work.J[1, col_j]) * work.w[col_j + 1]
                )
                work.w[col_j + 1] = -work.J[1, col_j] * tmp + work.J[0, col_j] * work.w[col_j + 1]
            if j < n:
                rho = np.sqrt(
                    np.conj(work.w[j]) * work.w[j] + np.conj(work.w[j + 1]) * work.w[j + 1]
                )
                work.J[0, j] = work.w[j] / rho
                work.J[1, j] = work.w[j + 1] / rho
                work.y[j + 1] = -work.J[1, j] * work.y[j]
                work.y[j] *= np.conj(work.J[0, j])
                work.w[j] = rho
            work.R[: j + 1, j] = work.w[: j + 1]
            # Estimate the absolute condition number
            kappa, inrm, nrm = _est_abs_cond(work.R, j, inrm, nrm, inrm_buf)
            if it >= maxit:
                flag = 1
                break
            it += 1
            if j + 1 >= restart:
                break
            err = rtol * 1e10  # just a large number
            if kappa >= form_x_thres:
                # Explicitly form x
                work.u2[:] = x
                work.w2[: j + 1] = work.y[: j + 1]
                _form_x(work.R, j, work.w2, work.Z, work.u2)
                mul_ax_func(A.indptr, A.indices, A.data, work.u2, work.w, trans=leftnull)
                err = np.linalg.norm(work.w, ord=1) / (A_1nrm * np.linalg.norm(work.u2, ord=1))
                if err <= rtol:
                    break
                if err >= null_res_prev:
                    j -= 1
                    flag = 3
                    ref_iters -= ref_its
                    err = null_res_prev
                    break
                null_res_prev = err
            j += 1
        # Inf loop
        _form_x(work.R, j, work.y, work.Z, x)
        if err <= rtol or flag:
            break
    x /= np.linalg.norm(x)
    return x, flag, it, ref_iters


def _est_abs_cond(
    R: np.ndarray, i: int, inrm: float, nrm: float, buf: np.ndarray
) -> typing.Tuple[float, float, float]:
    """Estimate the abs condition number"""
    if i == 0:
        buf[0] = 1.0 / R[0, 0]
        inrm = nrm = abs(buf[0])
    else:
        s = np.dot(buf[:i], R[:i, i])
        k1 = 1.0 - s
        k2 = -1.0 - s
        buf[i] = k2 / R[i, i] if abs(k1) < abs(k2) else k1 / R[i, i]
        inrm = max(inrm, abs(buf[i]))
        nrm = max(nrm, np.linalg.norm(R[: i + 1, i], ord=1))
    return inrm * nrm, inrm, nrm


def _form_x(R: np.ndarray, j: int, y: np.ndarray, Z: np.ndarray, x: np.ndarray) -> None:
    """Explicitly form the solution x"""
    la.solve_triangular(R[: j + 1, : j + 1], y[: j + 1], lower=False, overwrite_b=True)
    for i in range(j + 1):
        x += y[i] * Z[i, :]
