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
"""Utilities"""
import sys
import scipy.sparse

if sys.version_info >= (3, 9):
    Tuple = tuple
    List = list
    Dict = dict
else:
    import typing

    Tuple = typing.Tuple
    List = typing.List
    Dict = typing.Dict
    del typing
del sys


def to_crs(A: scipy.sparse.spmatrix) -> scipy.sparse.csr_matrix:
    """Helper function to convert to CRS"""
    if not scipy.sparse.issparse(A):
        raise TypeError("Must be SciPy sparse matrix")
    if not scipy.sparse.isspmatrix_csr(A):
        return A.tocsr()
    return A


to_csr = to_crs


def must_1d(x) -> None:
    """Helper function to ensure array must be 1D"""
    if (len(x.shape) > 1 and x.shape[1] != 1) or len(x.shape) > 2:
        raise ValueError("Must be 1D array")


def ensure_same(a, b, msg=None) -> None:
    """Helper to ensure same length (or values)"""
    if a != b:
        if msg is None:
            msg = "Unmatched sizes"
        raise ValueError("{} {} != {}".format(msg, a, b))
