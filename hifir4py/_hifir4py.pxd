# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
#                                                                             #
#    Copyright (C) 2019--2021 NumGeom Group at Stony Brook University         #
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU Affero General Public License as published #
#    by the Free Software Foundation, either version 3 of the License, or     #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU Affero General Public License for more details.                      #
#                                                                             #
#    You should have received a copy of the GNU Affero General Public License #
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.   #
###############################################################################

# Authors:
#   Qiao,

from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.memory cimport shared_ptr
cimport hifir4py as hif


ctypedef shared_ptr[hif.PyHIF] PyHIF_ptr
ctypedef shared_ptr[hif.PyHIF_Mixed] PyHIF_Mixed_ptr
ctypedef fused M_type:
    PyHIF_ptr
    PyHIF_Mixed_ptr
ctypedef shared_ptr[hif.KspSolver] KSP_ptr


# factorize
cdef inline void factorize_M(
    M_type M,
    size_t n,
    int[::1] rowptr,
    int[::1] colind,
    double[::1] vals,
    size_t m,
    hif.Options *opts,
) nogil:
    M.get().factorize_raw(n, &rowptr[0], &colind[0], &vals[0], m, opts[0])


# solve
cdef inline void solve_M(
    M_type M,
    size_t n,
    double[::1] b,
    double[::1] x,
    bool trans,
    size_t r,
) nogil:
    M.get().solve_raw(n, &b[0], &x[0], trans, r)


# solve with iterative refinement
cdef inline void solve_M_IR(
    M_type M,
    size_t n,
    int[::1] rowptr,
    int[::1] colind,
    double[::1] vals,
    double[::1] b,
    size_t N,
    double[::1] x,
    bool trans,
    size_t r,
) nogil:
    M.get().hifir_raw(n, &rowptr[0], &colind[0], &vals[0], &b[0], N, &x[0],
        trans, r)


# solve
cdef inline void mmultiply_M(
    M_type M,
    size_t n,
    double[::1] b,
    double[::1] x,
    bool trans,
    size_t r,
) nogil:
    M.get().mmultiply_raw(n, &b[0], &x[0], trans, r)


# ksp solve
cdef inline pair[int, size_t] solve_KSP(
    KSP_ptr solver,
    size_t n,
    int[::1] rowptr,
    int[::1] colind,
    double[::1] vals,
    double[::1] b,
    double[::1] x,
    int kernel,
    bool with_init_guess,
    bool verbose
) nogil:
    return solver.get().solve_raw(
        n,
        &rowptr[0],
        &colind[0],
        &vals[0],
        &b[0],
        &x[0],
        kernel,
        with_init_guess,
        verbose,
    )
