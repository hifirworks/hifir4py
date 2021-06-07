# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
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

# Authors:
#   Qiao,

# This is the core interface for hifir4py

from libcpp cimport bool
from libcpp.string cimport string as std_string
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.utility cimport pair
from cpython.ref cimport PyObject


cdef extern from "hifir4py.hpp" namespace "hif":
    # nullspace eliminator with gil
    cdef cppclass PyNspFilter:
        PyNspFilter()
        PyNspFilter(const size_t start)
        PyNspFilter(const size_t start, const size_t end)
        object (*array_encoder)(void *, size_t)
        void (*nsp_invoker)(void *, object)
        PyObject *user_call
        void enable_or()


cdef extern from "hifir4py.hpp" namespace "hif" nogil:
    # two necessary utilities
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

    ctypedef struct Options:
        pass
    Options get_default_options()
    std_string opt_repr(const Options &opts) except +
    # also, manipulation method
    bool set_option_attr[T](const std_string &attr, const T v, Options &opts)
    # enable verbose flags
    void enable_verbose(const int flag, Options &opts)
    # get verbose
    std_string get_verbose(const Options &opts);

    # io
    void read_hifir(const std_string &fn, size_t &nrows, size_t &ncols,
                    size_t &m, vector[int] &indptr,
                    vector[int] &indices, vector[double] &vals,
                    const bool is_bin) except +
    void write_hifir(const std_string &fn, const size_t nrows,
                     const size_t ncols, const int *indptr,
                     const int *indices, const double *vals,
                     const size_t m0, const bool is_bin) except +
    void query_hifir_info(const std_string &fn, bool &is_row, bool &is_c,
                          bool &is_double, bool &is_real,
                          uint64_t &nrows, uint64_t &ncols,
                          uint64_t &nnz, uint64_t &m,
                          const bool is_bin) except +

    cdef cppclass PyHIF:
        PyHIF()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t last_rank()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize_raw(const size_t n, const int *indptr, const int *indices,
                           const double *vals, const size_t m0,
                           const Options &opts) except +

        # solving routine
        void solve_raw(const size_t n, const double *b, double *x,
                       const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir_raw(const size_t n, const int *rowptr, const int *colind,
                       const double *vals, const double *b, const size_t N,
                       double *x, const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply_raw(const size_t n, const double *b, double *x,
                           const bool trans, const size_t r) except +

        shared_ptr[PyNspFilter] nsp

    cdef cppclass PyHIF_Mixed:
        PyHIF_Mixed()
        bool empty()
        size_t levels()
        size_t nnz()
        size_t nnz_ef()
        size_t nnz_ldu()
        size_t nrows()
        size_t ncols()
        size_t rank()
        size_t last_rank()
        size_t stats(const size_t entry) except +
        void clear() except +

        # computing routine
        void factorize_raw(const size_t n, const int *indptr, const int *indices,
                           const double *vals, const size_t m0,
                           const Options &opts) except +

        # solving routine
        void solve_raw(const size_t n, const double *b, double *x,
                       const bool trans, const size_t r) except +

        # solving with iterative refinement
        void hifir_raw(const size_t n, const int *rowptr, const int *colind,
                       const double *vals, const double *b, const size_t N,
                       double *x, const bool trans, const size_t r) except +

        # multilevel matrix-vector
        void mmultiply_raw(const size_t n, const double *b, double *x,
                           const bool trans, const size_t r) except +

        shared_ptr[PyNspFilter] nsp

    cdef cppclass KspSolver:
        void set_rtol(const double v)
        double get_rtol()
        void set_restart(const int v)
        int get_restart()
        void set_maxit(const size_t v)
        size_t get_maxit()
        void set_inner_steps(const size_t v)
        size_t get_inner_steps()
        void set_lamb1(const double v)
        double get_lamb1()
        void set_lamb2(const double v)
        double get_lamb2()
        void check_pars()
        int get_resids_length()
        void get_resids(double *r)
        pair[int, size_t] solve_raw(const size_t n, const int *rowptr,
                                    const int *colind, const double *vals,
                                    const double *b, double *x, const int kernel,
                                    const bool with_init_guess,
                                    const bool verbose) except +

    cdef cppclass PyGMRES(KspSolver):
        PyGMRES()
        PyGMRES(shared_ptr[PyHIF] M, const double rel_tol, const int rs,
                const size_t max_iters, const size_t max_inner_steps) except +
        void set_M(shared_ptr[PyHIF] M) except +
        shared_ptr[PyHIF] get_M()

    cdef cppclass PyGMRES_Mixed(KspSolver):
        PyGMRES_Mixed()
        PyGMRES_Mixed(shared_ptr[PyHIF_Mixed] M, const double rel_tol,
                      const int rs, const size_t max_iters,
                      const size_t max_inner_steps) except +
        void set_M(shared_ptr[PyHIF_Mixed] M) except +
        shared_ptr[PyHIF_Mixed] get_M()

    cdef cppclass PyFQMRCGSTAB(KspSolver):
        PyFQMRCGSTAB()
        PyFQMRCGSTAB(shared_ptr[PyHIF] M, const double rel_tol,
                 const size_t max_iters, const size_t innersteps) except +
        void set_M(shared_ptr[PyHIF] M) except +
        shared_ptr[PyHIF] get_M()

    cdef cppclass PyFQMRCGSTAB_Mixed(KspSolver):
        PyFQMRCGSTAB_Mixed()
        PyFQMRCGSTAB_Mixed(shared_ptr[PyHIF_Mixed] M, const double rel_tol,
                           const size_t max_iters,
                           const size_t innersteps) except +
        void set_M(shared_ptr[PyHIF_Mixed] M) except +
        shared_ptr[PyHIF_Mixed] get_M()

    cdef cppclass PyFBICGSTAB(KspSolver):
        PyFBICGSTAB()
        PyFBICGSTAB(shared_ptr[PyHIF] M, const double rel_tol,
                 const size_t max_iters, const size_t innersteps) except +
        void set_M(shared_ptr[PyHIF] M) except +
        shared_ptr[PyHIF] get_M()

    cdef cppclass PyFBICGSTAB_Mixed(KspSolver):
        PyFBICGSTAB_Mixed()
        PyFBICGSTAB_Mixed(shared_ptr[PyHIF_Mixed] M, const double rel_tol,
                           const size_t max_iters,
                           const size_t innersteps) except +
        void set_M(shared_ptr[PyHIF_Mixed] M) except +
        shared_ptr[PyHIF_Mixed] get_M()

    cdef cppclass PyTGMRESR(KspSolver):
        PyTGMRESR()
        PyTGMRESR(shared_ptr[PyHIF] M, const double rel_tol,
                 const size_t max_iters, const size_t innersteps) except +
        void set_M(shared_ptr[PyHIF] M) except +
        shared_ptr[PyHIF] get_M()

    cdef cppclass PyTGMRESR_Mixed(KspSolver):
        PyTGMRESR_Mixed()
        PyTGMRESR_Mixed(shared_ptr[PyHIF_Mixed] M, const double rel_tol,
                           const size_t max_iters,
                           const size_t innersteps) except +
        void set_M(shared_ptr[PyHIF_Mixed] M) except +
        shared_ptr[PyHIF_Mixed] get_M()


cdef extern from "hifir4py.hpp" namespace "hif::ksp" nogil:
    cdef enum:
        INVALID_ARGS
        M_SOLVE_ERROR
        SUCCESS
        DIVERGED
        STAGNATED
        BREAK_DOWN

    cdef enum:
        TRADITION
        ITERATIVE_REFINE
        CHEBYSHEV_ITERATIVE_REFINE
