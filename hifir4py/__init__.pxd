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

# Authors:
#   Qiao,

# This is the core interface for hifir4py

from libcpp cimport bool
from libcpp.string cimport string as std_string
from libcpp.utility cimport pair
from libc.stddef cimport size_t
from libc.stdint cimport int64_t


cdef extern from "hifir4py.hpp" namespace "hif" nogil:
    # Two utilities
    std_string version()
    bool warn_flag(const int)
    # wrap options, we don't care about the attributes
    cdef enum:
        VERBOSE_NONE
        VERBOSE_INFO
        VERBOSE_PRE
        VERBOSE_FAC
        VERBOSE_PRE_TIME
        VERBOSE_MEM
    cdef enum:
        REORDER_OFF
        REORDER_AUTO
        REORDER_AMD
        REORDER_RCM
    cdef enum:
        PIVOTING_OFF
        PIVOTING_ON
        PIVOTING_AUTO


cdef extern from "hifir4py.hpp" namespace "hifir4py" nogil:
    cdef enum:
        NUM_PARAMS

    void set_default_params(double *params)

    cdef cppclass di32PyHIF:
        ctypedef int index_type
        ctypedef double interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        di32PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +

    cdef cppclass di64PyHIF:
        ctypedef int64_t index_type
        ctypedef double interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        di64PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +

    cdef cppclass si32PyHIF:
        ctypedef int index_type
        # input is always double precision
        ctypedef double interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        si32PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +

    cdef cppclass si64PyHIF:
        ctypedef int64_t index_type
        # input is always double precision
        ctypedef double interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        si64PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +

    cdef cppclass zi32PyHIF:
        ctypedef int index_type
        ctypedef void interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        zi32PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +

    cdef cppclass zi64PyHIF:
        ctypedef int64_t index_type
        ctypedef void interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        zi64PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +

    cdef cppclass ci32PyHIF:
        ctypedef int index_type
        ctypedef void interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        ci32PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +

    cdef cppclass ci64PyHIF:
        ctypedef int64_t index_type
        ctypedef void interface_type

        @staticmethod
        bool is_mixed()
        @staticmethod
        bool is_complex()
        @staticmethod
        size_t index_size()

        ci64PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t schur_rank()
        size_t schur_size()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const double *params) except +

        # solving routine
        void solve(const size_t n, const interface_type *b, interface_type *x,
            const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply(const size_t n, const interface_type *b,
            interface_type *x, const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, interface_type *x,
            const bool trans, const size_t r) except +

        # solving with iterative refinement and residual bounds
        pair[size_t, int] hifir(const size_t n, const index_type *rowptr,
            const index_type *colind, const interface_type *vals,
            const interface_type *b, const size_t nirs, const double *betas,
            interface_type *x, const bool trans, const size_t r) except +
