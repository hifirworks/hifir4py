# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HILUCSI4PY project                     #
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
"""Utility module for ``hilucsi4py``

This module contains some helpful utility routines for ``hilucsi4py``, the core
functions are

1. parsing user arbitrary inputs, and
2. ensuring the data types

Notice that we use ``int`` as index type and ``double`` as floating number type,
thus the corresponding numpy dtypes are ``intc`` and ``float64``.

.. module:: hilucsi4py.utils
    :noindex:
.. moduleauthor:: Qiao Chen <qiao.chen@stonybrook.edu>
"""

import numpy as np

__all__ = ["convert_to_crs", "convert_to_crs_and_b"]


def _convert_to_crs(*args, shape=None):
    """Given user inputs, convert them to proper CRS three arrays with size
    """
    if len(args) != 3 and len(args) != 1:
        raise TypeError(
            "input matrix must be CRS's three arrays or scipy sparse matrix"
        )
    is_3_arr = len(args) == 3
    if is_3_arr and shape is None:
        raise ValueError("shape is missing")
    if is_3_arr:
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError("shape must be tuple of dimension 2")
        if shape[0] != shape[1]:
            raise ValueError("the matrix must be squared")
        return args[0], args[1], args[2], shape[0]
    if not hasattr(args[0], "tocsr"):
        raise TypeError("single-argument input must be scipy sparse matrix")
    A = args[0].tocsr()
    if shape is not None and shape != A.shape:
        raise ValueError(
            "inconsistent user-shape {} and matrix shape {}".format(shape, A.shape)
        )
    shape = A.shape
    if shape[0] != shape[1]:
        raise ValueError("the matrix must be squared")
    return A.indptr, A.indices, A.data, shape[0]


def _as_index_array(v):
    return np.asarray(v, dtype=np.intc)


def _as_value_array(v):
    return np.asarray(v, dtype=np.float64)


def _is_binary(filename):
    # NOTE: try to detect binary file
    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
    with open(filename, "rb") as fin:
        flag = fin.read(1024).translate(None, textchars)
    return flag


def convert_to_crs(*args, shape=None):
    """Helper function to convert arbitrarily user inputs

    In Python, a flexible user interface should not restrict what data types
    the user must obey. At the same time, the code should also be efficient if
    the user provides the correct data types. Luckily, this can be easily done
    in numpy with the function ``asarray``.

    We allow two types of user inputs

    1. the user provides the typical three-array matrix representation
    2. the user provides a class that follows scipy sparse design.

    For the latter, we essentially just require the input matrix has a member
    function of ``tocsr``, which returns an object that has attributes of
    ``indptr``, ``indices``, and ``vals``, which can be further converted to
    numpy arrays with data types ``int``, ``int``, and ``double``, resp.

    Parameters
    ----------
    *args : positional arguments
        either three array of CRS or "scipy sparse matrix"-like
    shape : tuple, optional
        if input is three array, then this must be given

    Returns
    -------
    rowptr : np.ndarray
        compressed row starting position array
    colind : np.ndarray
        Column index array
    vals : np.ndarray
        Data array

    See Also
    --------
    convert_to_crs_and_b
    """
    rowptr, colind, vals, n = _convert_to_crs(*args, shape=shape)
    rowptr = _as_index_array(rowptr)
    assert len(rowptr.shape) == 1
    assert rowptr.size == n + 1, "invalid rowptr size"
    colind = _as_index_array(colind)
    assert len(colind.shape) == 1
    vals = _as_value_array(vals)
    assert len(vals.shape) == 1
    assert colind.size == vals.size, "colind and vals should have same size"
    assert rowptr[n] == colind.size, "inconsistent rowptr[n] and colind.size"
    return rowptr, colind, vals


def convert_to_crs_and_b(*args, shape=None):
    """Helper function to convert arbitrarily user inputs with rhs

    Same functionality as :func:`convert_to_crs` with additional entry of rhs
    vector `b`.
    """
    last = len(args) - 1
    if not last:
        raise ValueError("invalid inputs")
    rowptr, colind, vals = convert_to_crs(*args[:last], shape=shape)
    b = _as_value_array(args[last])
    if b.size != len(rowptr) - 1:
        raise ValueError("rhs must have size of n ({})".format(len(rowptr) - 1))
    return rowptr, colind, vals, b.reshape(-1)
