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

# Generic implementation of PyHIF
# Author(s):
#   Qiao Chen

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libc.stddef cimport size_t


cdef class HIF:
    cdef ptr_t[prec_t] M

    @staticmethod
    def is_mixed():
        return prec_t.is_mixed()

    @staticmethod
    def is_complex():
        return prec_t.is_complex()

    @staticmethod
    def index_size():
        return prec_t.index_size()

    def __cinit__(self):
        self.M.reset(new prec_t())

    def empty(self):
        """Check if or not the builder is empty"""
        return deref(self.M).empty()

    @property
    def levels(self):
        """int: number of levels"""
        return deref(self.M).levels()

    @property
    def nnz(self):
        """int: total number of nonzeros of all levels"""
        return deref(self.M).nnz()

    @property
    def nnz_ef(self):
        """int: total number of nonzeros in Es and Fs"""
        return deref(self.M).nnz_ef()

    @property
    def nnz_ldu(self):
        """int: total number of nonzeros in LDU fators"""
        return deref(self.M).nnz_ldu()

    @property
    def nrows(self):
        """int: system size of rows"""
        return deref(self.M).nrows()

    @property
    def ncols(self):
        """int: system size of columns"""
        return deref(self.M).ncols()

    @property
    def rank(self):
        """int: numerical rank"""
        return deref(self.M).rank()

    @property
    def schur_rank(self):
        """int: final Schur complement rank"""
        return deref(self.M).schur_rank()

    @property
    def schur_size(self):
        """int: final Schur complement size"""
        return deref(self.M).schur_size()

    def stats(self, int entry):
        """Get the statistics information"""
        return deref(self.M).stats(entry)

    def factorize(
        self,
        index_t[::1] rowptr,
        index_t[::1] colind,
        value_t[::1] vals,
        double[::1] params
    ):
        """Low-level factorization"""
        cdef size_t n = rowptr.size - 1
        deref(self.M).factorize(n, &rowptr[0], &colind[0], &vals[0], &params[0])

    def solve(self, value_t[::1] b, value_t[::1] x, bool trans, size_t r):
        """Low-level multilevel triangular solve"""
        cdef size_t n = b.size
        deref(self.M).solve(n, &b[0], &x[0], trans, r)

    def mmultiply(self, value_t[::1] b, value_t[::1] x, bool trans, size_t r):
        """Low-level multilevel matrix-vector multiplication"""
        cdef size_t n = b.size
        deref(self.M).mmultiply(n, &b[0], &x[0], trans, r)

    def hifir(
        self,
        index_t[::1] rowptr,
        index_t[::1] colind,
        value_t[::1] vals,
        value_t[::1] b,
        size_t nirs,
        value_t[::1] x,
        bool trans,
        size_t r,
    ):
        """Low-level triangular solve with iterative refinements"""
        cdef size_t n = rowptr.size - 1
        deref(self.M).hifir(
            n, &rowptr[0], &colind[0], &vals[0], &b[0], nirs, &x[0], trans, r
        )

    def hifir_beta(
        self,
        index_t[::1] rowptr,
        index_t[::1] colind,
        value_t[::1] vals,
        value_t[::1] b,
        size_t nirs,
        double[::1] betas,
        value_t[::1] x,
        bool trans,
        size_t r,
    ):
        """Low-level triangular solve with IR and residual bounds"""
        cdef size_t n = rowptr.size - 1
        return deref(self.M).hifir(
            n, &rowptr[0], &colind[0], &vals[0], &b[0], nirs, &betas[0], &x[0], trans, r
        )
