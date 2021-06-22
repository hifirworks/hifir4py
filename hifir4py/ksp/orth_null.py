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

import numpy as np
from .gmres import GMRES_WorkSpace

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


def orth_null():
    pass


def _fgmres_null(  # noqa: C901
    A,
    b,
    M,
    leftnull: bool,
    restart: int,
    rtol: float,
    maxit: int,
    x,
    zero_start: bool,
    work: FGMRES_WorkSpace,
    mul_ax_func,
):
    """Single nullspace solver using customized HIFIR-FGMRES with Householder"""
    import scipy.sparse.linalg as spla
    import scipy.linalg as la

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
                work.v -= 2.0 * np.vdot(work.Q[i, :], work.v) * work.Q[i, :]
            work.v /= np.linalg.norm(work.v)
            work.u2[:] = work.v
            ref_its = _iter_refine(
                A,
                M,
                nirs,
                work.u2,
                work.v,  # output
                work.w,
                work.u,
                leftnull,
                mul_ax_func,
                fgmres_iter=it,
                bnorm=1.0,
            )
            ref_iters += ref_its
            work.Z[j, :] = work.v
            mul_ax_func(A.indptr, A.indices, A.data, work.v, work.w, trans=leftnull)

            # Orthogonalize the Krylov vector
            for i in range(j + 1):
                work.w -= 2.0 * np.vdot(work.Q[i, :], work.w) * work.Q[i, :]
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
                        updated_norm = np.sqrt(
                            2.0 * alpha2 + 2.0 * np.real(work.u[j + 1]) * alpha
                        )
                        work.u[j + 1] += alpha
                        work.Q[j + 1, j + 1 :] = work.u[j + 1 :] / updated_norm
                    work.w[j + 2 :] = 0.0
                    work.w[j + 1] = -alpha

            # Apply Given's rotation
            # Given's rotation
            for col_j in range(j):
                tmp = work.w[col_j]
                work.w[col_j] = (
                    np.conj(work.J[0, col_j]) * tmp
                    + np.conj(work.J[1, col_j]) * work.w[col_j + 1]
                )
                work.w[col_j + 1] = (
                    -work.J[1, col_j] * tmp + work.J[0, col_j] * work.w[col_j + 1]
                )
            if j < n:
                rho = np.sqrt(
                    np.conj(work.w[j]) * work.w[j]
                    + np.conj(work.w[j + 1]) * work.w[j + 1]
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
                y2 = la.solve_triangular(
                    work.R[: j + 1, : j + 1], work.y[: j + 1], lower=False
                )
                for i in range(j + 1):
                    work.u2 += y2[i] * work.Z[i, :]
                mul_ax_func(
                    A.indptr, A.indices, A.data, work.u2, work.w, trans=leftnull
                )
                err = np.linalg.norm(work.w, ord=1) / (
                    A_1nrm * np.linalg.norm(work.u2, ord=1)
                )
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
        la.solve_triangular(
            work.R[: j + 1, : j + 1], work.y[: j + 1], lower=False, overwrite_b=True
        )
        for i in range(j + 1):
            x += work.y[i] * work.Z[i, :]
        if err <= rtol or flag:
            break
    if flag == 0:
        x /= np.linalg.norm(x)
    return x, flag, it, ref_iters


def _iter_refine(
    A,
    M,
    nirs,
    b,
    x,
    w,
    r,
    leftnull,
    mul_ax_func,
    *,
    fgmres_iter=-1,
    beta_L=0.2,
    beta_U=10.0,
    bnorm=None,
):
    """Customized iterative refinement"""
    if fgmres_iter == 0:
        nirs = min(nirs, 4)
    if bnorm is None:
        bnorm = np.linalg.norm(b)
    x[:] = 0.0
    r[:] = b
    op = "S" if not leftnull else "SH"
    it = 0
    while True:
        w = M.apply(r, op=op, x=w, rank=-1)
        x += w
        mul_ax_func(A.indptr, A.indices, A.data, x, w, trans=leftnull)
        r[:] = b - w
        res = np.linalg.norm(r) / bnorm
        it += 1
        if it >= nirs or res > beta_U or res <= beta_L:
            break
    return it


def _est_abs_cond(R, i, inrm, nrm, buf):
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
