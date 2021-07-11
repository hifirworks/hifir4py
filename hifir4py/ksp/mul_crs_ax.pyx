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

# Matrix-vector multiplication for CRS

from libcpp cimport bool
from libc.stddef cimport size_t
cimport hifir4py

ctypedef hifir4py.int64_t int64_t


cdef extern from "hifir4py.hpp" namespace "hif" nogil:
    cdef cppclass CRS[V, I]:
        CRS()
        void multiply_t_low(const V *x, V *y)

    CRS[V, I] wrap_const_crs[V, I](const size_t nrows, const size_t ncols,
        const I *row_start, const I *col_ind, const V *vals, bool check)


cdef extern from "hifir4py.hpp" namespace "hif::mt" nogil:
    # Multi-threaded CRS matrix-vector
    void multiply_nt_low[Mat, V](const Mat &A, const V *x, V *y)


ctypedef CRS[double, int] matdi32_t
ctypedef CRS[double, int64_t] matdi64_t
ctypedef CRS[double complex, int] matzi32_t
ctypedef CRS[double complex, int64_t] matzi64_t


##################
# double int32
##################

cdef inline void _di32_multiply(size_t n, int[::1] indptr, int[::1] indices,
    double[::1] vals, double[::1] x, double[::1] y) nogil:
    cdef matdi32_t A = wrap_const_crs[double, int](n, n, &indptr[0],
        &indices[0], &vals[0])
    multiply_nt_low[matdi32_t, double](A, &x[0], &y[0])


cdef inline void _di32_multiply_trans(size_t n, int[::1] indptr, int[::1] indices,
    double[::1] vals, double[::1] x, double[::1] y) nogil:
    cdef matdi32_t A = wrap_const_crs[double, int](n, n, &indptr[0],
        &indices[0], &vals[0])
    A.multiply_t_low(&x[0], &y[0])


def di32_multiply(
    int[::1] indptr,
    int[::1] indices,
    double[::1] vals,
    double[::1] x,
    double[::1] y,
    bool trans=False,
):
    """Matrix-vector of y=A*x for double and int32"""
    if not trans:
        _di32_multiply(indptr.size - 1, indptr, indices, vals, x, y)
    else:
        _di32_multiply_trans(indptr.size - 1, indptr, indices, vals, x, y)


##################
# double int64
##################

cdef inline void _di64_multiply(size_t n, int64_t[::1] indptr, int64_t[::1] indices,
    double[::1] vals, double[::1] x, double[::1] y) nogil:
    cdef matdi64_t A = wrap_const_crs[double, int64_t](n, n, &indptr[0],
        &indices[0], &vals[0])
    multiply_nt_low[matdi64_t, double](A, &x[0], &y[0])


cdef inline void _di64_multiply_trans(size_t n, int64_t[::1] indptr,
    int64_t[::1] indices, double[::1] vals, double[::1] x, double[::1] y) nogil:
    cdef matdi64_t A = wrap_const_crs[double, int64_t](n, n, &indptr[0],
        &indices[0], &vals[0])
    A.multiply_t_low(&x[0], &y[0])


def di64_multiply(
    int64_t[::1] indptr,
    int64_t[::1] indices,
    double[::1] vals,
    double[::1] x,
    double[::1] y,
    bool trans=False,
):
    """Matrix-vector of y=A*x for double and int64"""
    if not trans:
        _di64_multiply(indptr.size - 1, indptr, indices, vals, x, y)
    else:
        _di64_multiply_trans(indptr.size - 1, indptr, indices, vals, x, y)


#######################
# complex double int32
#######################

ctypedef double complex z_t


cdef inline void _zi32_multiply(size_t n, int[::1] indptr, int[::1] indices,
    z_t[::1] vals, z_t[::1] x, z_t[::1] y) nogil:
    cdef matzi32_t A = wrap_const_crs[z_t, int](n, n, &indptr[0],
        &indices[0], &vals[0])
    multiply_nt_low[matzi32_t, z_t](A, &x[0], &y[0])


cdef inline void _zi32_multiply_trans(size_t n, int[::1] indptr, int[::1] indices,
    z_t[::1] vals, z_t[::1] x, z_t[::1] y) nogil:
    cdef matzi32_t A = wrap_const_crs[z_t, int](n, n, &indptr[0],
        &indices[0], &vals[0])
    A.multiply_t_low(&x[0], &y[0])


def zi32_multiply(
    int[::1] indptr,
    int[::1] indices,
    z_t[::1] vals,
    z_t[::1] x,
    z_t[::1] y,
    bool trans=False,
):
    """Matrix-vector of y=A*x for double complex and int32"""
    if not trans:
        _zi32_multiply(indptr.size - 1, indptr, indices, vals, x, y)
    else:
        _zi32_multiply_trans(indptr.size - 1, indptr, indices, vals, x, y)


#######################
# complex double int64
#######################

cdef inline void _zi64_multiply(size_t n, int64_t[::1] indptr, int64_t[::1] indices,
    z_t[::1] vals, z_t[::1] x, z_t[::1] y) nogil:
    cdef matzi64_t A = wrap_const_crs[z_t, int64_t](n, n, &indptr[0],
        &indices[0], &vals[0])
    multiply_nt_low[matzi64_t, z_t](A, &x[0], &y[0])


cdef inline void _zi64_multiply_trans(size_t n, int64_t[::1] indptr,
    int64_t[::1] indices, z_t[::1] vals, z_t[::1] x, z_t[::1] y) nogil:
    cdef matzi64_t A = wrap_const_crs[z_t, int64_t](n, n, &indptr[0],
        &indices[0], &vals[0])
    A.multiply_t_low(&x[0], &y[0])


def zi64_multiply(
    int64_t[::1] indptr,
    int64_t[::1] indices,
    z_t[::1] vals,
    z_t[::1] x,
    z_t[::1] y,
    bool trans=False,
):
    """Matrix-vector of y=A*x for double complex and int64"""
    if not trans:
        _zi64_multiply(indptr.size - 1, indptr, indices, vals, x, y)
    else:
        _zi64_multiply_trans(indptr.size - 1, indptr, indices, vals, x, y)
