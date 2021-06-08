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
"""This is the main module contains the implementation of ``hifir4py``

The module wraps the internal components defined in HIF and safely brings
them in Python3. This module includes:

1. multilevel preconditioner,
2. control parameters,
3. KSP solver(s)
4. IO with native HIF binary and ASCII files

.. module:: hifir4py._hifir4py
    :noindex:
.. moduleauthor:: Qiao Chen <qiao.chen@stonybrook.edu>
"""

from cython.operator cimport dereference as deref
from cpython.ref cimport PyObject, Py_XINCREF
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from libcpp.utility cimport pair
cimport hifir4py as hif


cdef extern from "hifir4py.hpp" namespace 'hif::internal' nogil:
    # using an internal var to determine the data types of options
    # true for double, flase for int
    cdef enum:
        _HIF_TOTAL_OPTIONS
    bool option_dtypes[_HIF_TOTAL_OPTIONS]


ctypedef hif.PyGMRES *gmres_ptr
ctypedef hif.PyGMRES_Mixed *gmres_mixed_ptr
ctypedef hif.PyFGMRES *fgmres_ptr
ctypedef hif.PyFGMRES_Mixed *fgmres_mixed_ptr
ctypedef hif.PyFQMRCGSTAB *fqmrcgstab_ptr
ctypedef hif.PyFQMRCGSTAB_Mixed *fqmrcgstab_mixed_ptr
ctypedef hif.PyFBICGSTAB *fbicgstab_ptr
ctypedef hif.PyFBICGSTAB_Mixed *fbicgstab_mixed_ptr


import os
import numpy as np
from .utils import (
    convert_to_crs,
    convert_to_crs_and_b,
    _as_index_array,
    _as_value_array,
    _is_binary,
)


# convert raw data to numpy array (wrapping data), used in nullspace filter
cdef inline raw2ndarray(void *x, size_t n):
    cdef double *dx = <double *> x
    return np.asarray(<double[:n]> dx)


# call Python function, used in nullspace filter
cdef inline void call_user(void *f, object x):
    (<object>f)(x)


# whenever added new external objects, make sure update this table
__all__ = [
    "VERBOSE_NONE",
    "VERBOSE_INFO",
    "VERBOSE_PRE",
    "VERBOSE_FAC",
    "VERBOSE_PRE_TIME",
    "REORDER_OFF",
    "REORDER_AUTO",
    "REORDER_AMD",
    "REORDER_RCM",
    "PIVOTING_OFF",
    "PIVOTING_ON",
    "PIVOTING_AUTO",
    "Params",
    "read_hifir",
    "write_hifir",
    "query_hifir_info",
    "HIF",
    "HIF_Mixed",
    "KSP_Error",
    "KSP_InvalidArgumentsError",
    "KSP_MSolveError",
    "KSP_DivergedError",
    "KSP_StagnatedError",
    "KSP_BreakDownError",
    "KspSolver",
    "GMRES",
    "GMRES_Mixed",
    "FGMRES",
    "FGMRES_Mixed",
    "FQMRCGSTAB",
    "FQMRCGSTAB_Mixed",
    "FBICGSTAB",
    "FBICGSTAB_Mixed",
    "TGMRESR",
    "TGMRESR_Mixed",
]

# utilities
def _version():
    return hif.version().decode("utf-8")


def _is_warning():
    return hif.warn_flag(-1)


def _enable_warning():
    hif.warn_flag(1)


def _disable_warning():
    hif.warn_flag(0)


# Options
# redefine the verbose options, not a good idea but okay for now
VERBOSE_NONE = hif.VERBOSE_NONE
VERBOSE_INFO = hif.VERBOSE_INFO
VERBOSE_PRE = hif.VERBOSE_PRE
VERBOSE_FAC = hif.VERBOSE_FAC
VERBOSE_PRE_TIME = hif.VERBOSE_PRE_TIME

# reorderingoptions
REORDER_OFF = hif.REORDER_OFF
REORDER_AUTO = hif.REORDER_AUTO
REORDER_AMD = hif.REORDER_AMD
REORDER_RCM = hif.REORDER_RCM

# pivoting strategy
PIVOTING_OFF = hif.PIVOTING_OFF
PIVOTING_ON = hif.PIVOTING_ON
PIVOTING_AUTO = hif.PIVOTING_AUTO

# determine total number of parameters
def _get_opt_info():
    raw_info = hif.opt_repr(hif.get_default_options()).decode("utf-8")
    # split with newline
    info = list(filter(None, raw_info.split("\n")))
    return [x.split()[0].strip() for x in info]


_OPT_LIST = _get_opt_info()


cdef class Params:
    """Python interface of control parameters

    By default, each control parameter object is initialized with default
    values in the paper. In addition, modifying the parameters can be achieved
    by using key-value pairs, i.e. `__setitem__`. The keys are the names of
    those defined in original C/C++ ``struct``.

    Here is a complete list of parameter keys: ``tau_L``, ``tau_U``, ``tau_d``,
    ``tau_kappa``, ``alpha_L``, ``alpha_U``, ``rho``, ``c_d``, ``c_h``, ``N``,
    and ``verbose``. Please consult the original paper and/or the C++
    documentation for default information regarding these parameters.

    Examples
    --------

    >>> from hifir4py import *
    >>> params = Params()  # default parameters
    >>> params["verbose"] = VERBOSE_INFO | VERBOSE_FAC
    >>> params.reset()  # reset to default parameters
    """
    cdef hif.Options params
    cdef int __idx

    @staticmethod
    def param_list():
        """Get a supported parameter list

        Returns
        -------
        list(str)
            A list of option names
        """
        return _OPT_LIST.copy()

    def __init__(self, **kw):
        # for enabling docstring purpose
        pass

    def __cinit__(self, **kw):
        self.params = hif.get_default_options()
        self.__idx = 0
        for k, v in kw.items():
            self.__setitem__(k, v)

    def reset(self):
        """This function will reset all options to their default values"""
        self.params = hif.get_default_options()

    def enable_verbose(self, int flag):
        """Enable a verbose flag

        Parameters
        ----------
        flag : int
            enable a log flag, defined with variables starting with ``VERBOSE``
        """
        hif.enable_verbose(<int> flag, self.params)

    @property
    def verbose(self):
        """str: get the verbose flag(s)"""
        return hif.get_verbose(self.params).decode("utf-8")

    def __str__(self):
        return hif.opt_repr(self.params).decode("utf-8")

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, str opt_name, v):
        """Set a configuration with keyvalue pair

        Parameters
        ----------
        opt_name : str
            Option name
        v : int or float
            Corresponding value

        Raises
        ------
        KeyError
            Raised as per unsupported options
        """
        # convert to double
        cdef:
            double vv = v
            std_string nm = opt_name.encode("utf-8")
        if hif.set_option_attr[double](nm, vv, self.params):
            raise KeyError("unknown option name {}".format(opt_name))

    def __getitem__(self, str opt_name):
        """Retrieve the value given an option

        Parameters
        ----------
        opt_name : str
            Option name

        Returns
        -------
        int or float
            Option value
        """
        if opt_name not in _OPT_LIST:
            raise KeyError("unknown option name {}".format(opt_name))
        cdef int idx = _OPT_LIST.index(opt_name)
        attr = list(filter(None, self.__str__().split('\n')))[idx]
        v = list(filter(None, attr.split()))[1]
        if opt_name in ("check", "reorder", "verbose"):
            return v
        if option_dtypes[idx]:
            return float(v)
        return int(v)

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= len(_OPT_LIST):
            raise StopIteration
        try:
            return self.__getitem__(_OPT_LIST[self.__idx])
        finally:
            self.__idx += 1

    def dict(self):
        """Convert to dictionary
        """
        return {k: self.__getitem__(k) for k in _OPT_LIST}

    def keys(self):
        """Get keys
        """
        return self.dict().keys()

    def values(self):
        """Get values
        """
        return self.dict().values()

    def items(self):
        """Get items
        """
        return self.dict().items()


# I/O
def read_hifir(str filename, *, is_bin=None):
    """Read a HIFIR file

    Parameters
    ----------
    filename : str
        file name
    is_bin : bool, optional
        binary flag, leaving unset will automatically detect

    Returns
    -------
    indptr : np.ndarray
        Starting row position pointer array in CRS
    indices : np.ndarray
        List of column indices
    vals : np.ndarray
        List of numerical data values
    shape : tuple
        Matrix size shape, i.e., (nrows, ncols)
    m : int
        Size of leading symmetric block  (deprecated)

    See Also
    --------
    write_hifir : write native formats
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError
    cdef:
        std_string fn = filename.encode("utf-8")
        bool isbin
        vector[int] indptr
        vector[int] indices
        vector[double] vals
        size_t nrows = 0
        size_t ncols = 0
        size_t m = 0

    if is_bin is None:
        isbin = _is_binary(filename)
    else:
        isbin = <bool> is_bin
    hif.read_hifir(fn, nrows, ncols, m, indptr, indices, vals, isbin)
    return (
        _as_index_array(indptr),
        _as_index_array(indices),
        _as_value_array(vals),
        (nrows, ncols),
        m,
    )



cdef inline void _write_hifir(
    str filename,
    int[::1] rowptr,
    int[::1] colind,
    double[::1] vals,
    int nrows,
    int ncols,
    int m,
    bool is_bin
):
    cdef:
        std_string fn = filename.encode("utf-8")
        bool isbin = is_bin
    hif.write_hifir(fn, nrows, ncols, &rowptr[0], &colind[0], &vals[0],
        m, isbin)


def write_hifir(str filename, *args, shape=None, m=0, is_bin=True):
    """Write data to HIFIR file formats

    Parameters
    ----------
    filename : str
        file name
    *args : positional arguments
        either three array of CRS or scipy sparse matrix
    shape : ``None`` or tuple
        if input is three array, then this must be given
    m : int (optional)
        leading symmetric block (deprecated)
    is_bin : bool (optional)
        if ``True`` (default), then assume binary file format

    See Also
    --------
    read_hifir : read native formats
    """
    # essential checkings to avoid segfault
    cdef:
        size_t m0 = m
        bool isbin = is_bin
        size_t n
    rowptr, colind, vals = convert_to_crs(*args, shape=shape)
    assert len(rowptr), 'cannot write empty matrix'
    n = len(rowptr) - 1
    _write_hifir(filename, rowptr, colind, vals, n, n, m0, isbin)


def query_hifir_info(str filename, *, is_bin=None):
    """Read a HIFIR file and only query its information

    Parameters
    ----------
    filename : str
        file name
    is_bin : ``None`` or bool (optional)
        if ``None``, then will automatically detect

    Returns
    -------
    is_row : bool
        If or not the matrix in file is row major (CRS)
    is_c : bool
        If or not C-based (should be true)
    is_double : bool
        If or not double precision
    is_real : bool
        If or not real number
    nrows, ncols, nnz : int
        Matrix sizes
    m : int
        Leading symmetric block size
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError
    cdef:
        std_string fn = filename.encode("utf-8")
        bool isbin
        bool is_row
        bool is_c
        bool is_double
        bool is_real
        uint64_t nrows
        uint64_t ncols
        uint64_t nnz
        uint64_t m

    if is_bin is None:
        isbin = _is_binary(filename)
    else:
        isbin = <bool> is_bin
    hif.query_hifir_info(fn, is_row, is_c, is_double, is_real, nrows,
        ncols, nnz, m, isbin)
    return is_row, is_c, is_double, is_real, nrows, ncols, nnz, m


cdef class HIF:
    """Python HIF object

    The interfaces remain the same as the original user object, i.e.
    `hif::DefaultHIF`. However, we significantly the parameters by
    hiding the needs of `hif::CRS`, `hif::CCS`, and `hif::Array`.
    Therefore, the interface is very generic and easily adapted to any other
    Python modules without the hassle of complete object types.
    
    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> A = random(10, 10, 0.5)
    >>> M = HIF()
    >>> M.factorize(A)
    """
    cdef PyHIF_ptr M
    cdef hif.Options params

    @staticmethod
    def is_mixed():
        """Static method to check mixed precision
        """
        return False

    def __init__(self):
        # for docstring purpose
        pass

    def __cinit__(self):
        self.M.reset(new hif.PyHIF())
        self.params = hif.get_default_options()

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
        """Get the statistics information

        Parameters
        ----------
        entry: int
            entry field
        """
        return deref(self.M).stats(entry)

    def clear(self):
        """Clear internal memory usage"""
        deref(self.M).clear()

    def factorize(self, *args, shape=None, Params params=None):
        """Compute/build the preconditioner

        Parameters
        ----------
        *args : positional arguments
            either three array of CRS or scipy sparse matrix
        shape : tuple, optional
            if input is three arrays of CRS, then this must be given
        params : Params, optional
            control parameters, if ``None``, then use the default values

        See Also
        --------
        solve : solve for inv(HIF)*x
        """
        cdef:
            size_t n
            size_t m
        rowptr, colind, vals = convert_to_crs(*args, shape=shape)
        assert len(rowptr), "cannot deal with empty matrix"
        if params is not None:
            self.params = params.params
        n = len(rowptr) - 1
        m = 0
        factorize_M[PyHIF_ptr](self.M, n, rowptr, colind, vals, m, &self.params)

    def solve(self, b, x=None, trans=False, r=0):
        r"""Core routine to use the preconditioner

        Essentially, this routine is to perform
        :math:`\boldsymbol{x}=\boldsymbol{M}^{-1}\boldsymbol{b}`, where
        :math:`\boldsymbol{M}` is our MILU preconditioner.

        Parameters
        ----------
        b : array_like
            right-hand side parameters
        x : array_like, optional
            solution vector, output buffer
        trans : bool, optional
            transpose/Hermitian flag, default is False
        r : int, optional
            final Schur complement rank

        Returns
        -------
        np.ndarray
            Solution vector
        """
        cdef size_t n
        bb = _as_value_array(b)
        assert len(bb.shape) == 1
        if x is None:
            xx = np.empty_like(bb)
        else:
            xx = _as_value_array(x)
        assert xx.shape == bb.shape, "inconsistent x and b"
        n = bb.size
        solve_M[PyHIF_ptr](self.M, n, bb, xx, trans, r)
        return xx

    def hifir(
        self,
        *args,
        shape=None,
        N=2,
        x=None,
        trans=False,
        r=-1,
    ):
        """Access the preconditioner with iterative refinement

        Parameters
        ----------
        *args : positional arguments
            either three array of CRS or scipy sparse matrix, and rhs b
        shape : tuple, optional
            if input is three array, then this must be given
        N : int, optional
            explicit iteration steps, default is 2
        x : np.ndarray, optional
            for efficiency purpose, one can provide the buffer
        trans : bool, optional
            transpose/Hermitian flag, default is False
        r : int, optional
            final Schur complement rank

        Returns
        -------
        np.ndarray
            Solution vector
        """
        cdef:
            size_t n
            size_t NN
        rowptr, colind, vals, b = convert_to_crs_and_b(*args, shape=shape)
        if x is None:
            xx = np.empty_like(b)
        else:
            xx = _as_value_array(x)
        if xx.shape != b.shape:
            raise ValueError('inconsistent x and b')
        if N <= 0:
            N = 1
        n = b.size
        NN = N
        solve_M_IR[PyHIF_ptr](
            self.M,
            n,
            rowptr,
            colind,
            vals,
            b,
            NN,
            xx,
            trans,
            r,
        )
        return xx

    def mmultiply(self, b, x=None, trans=False, r=0):
        r"""Core routine to use the preconditioner for matrix-vector

        Essentially, this routine is to perform
        :math:`\boldsymbol{x}=\boldsymbol{M}\boldsymbol{b}`, where
        :math:`\boldsymbol{M}` is our MILU preconditioner.

        Parameters
        ----------
        b : array_like
            right-hand side parameters
        x : array_like, optional
            solution vector, output buffer
        trans : bool, optional
            transpose/Hermitian flag, default is False
        r : int, optional
            final Schur complement rank

        Returns
        -------
        np.ndarray
            Solution vector
        """
        cdef size_t n
        bb = _as_value_array(b)
        assert len(bb.shape) == 1
        if x is None:
            xx = np.empty_like(bb)
        else:
            xx = _as_value_array(x)
        assert xx.shape == bb.shape, "inconsistent x and b"
        n = bb.size
        mmultiply_M[PyHIF_ptr](self.M, n, bb, xx, trans, r)
        return xx

    def set_nsp_filter(self, *args):
        """Enable (right) null space filter

        One of the nice features in HIFIR (including HIFIR4PY) is to
        enabling (right) nullspace filter (eliminator) in preconditioner to
        solve singular systems.

        Parameters
        ----------
        arg1 : {None, int, callback}, optional
            First argument (later)
        arg2 : int, optional
            Second argument (later)

        Notes
        -----
        This function have the following usages:

        1. Set up a commonly used constant mode filter,
        2. Set up a constant mode filter within a certain range,
        3. Set up a filter via a user callback, and
        4. Unset the filter.

        For 1), this function takes no input argument. For 2), one or two
        arguments of integer can be passed, where the first one defines the
        starting index of the constant mode, whereas the pass-of-end position
        for the second one. For 3), the user needs to define a callback with
        a single input and output argument, where the input is guaranteed to
        be a 1D numpy ndarray, upon output, the user need to filter the null
        space components inplace to that vector. For 4), simply supplying
        ``None`` will remove the null space filter.

        Examples
        --------
        Given a preconditioner

        >>> M = create_M()

        We first consider the usage of 1)

        >>> M.set_nsp_filter()

        For 2)

        >>> M.set_nsp_filter(0)  # equiv as above
        >>> M.set_nsp_filter(0, 2)  # constant modes exist in [0, 2)

        For 3)

        >>> def my_filter(v):
        ...     v[:] -= np.sum(v) / v.size  # equiv as filter the constant mode
        ...
        >>> M.set_nsp_filter(my_filter)

        For 4)

        >>> M.set_nsp_filter(None)  # remove nullspace filter.
        """
        cdef:
            hif.PyNspFilter *py_nsp  # python null space filter
            size_t start
            size_t end
        if len(args) == 0:
            # simple constant mode
            deref(self.M).nsp.reset(new hif.PyNspFilter())
            return
        if args[0] is None:
            deref(self.M).nsp.reset()
            return
        if callable(args[0]):
            deref(self.M).nsp.reset(new hif.PyNspFilter())
            py_nsp = <hif.PyNspFilter *>deref(self.M).nsp.get()
            py_nsp.array_encoder = &raw2ndarray
            py_nsp.nsp_invoker = &call_user
            py_nsp.user_call = <PyObject*>args[0]
            Py_XINCREF(py_nsp.user_call)  # increment the reference
            py_nsp.enable_or()
            return
        start = args[0]
        end = <size_t>-1
        if len(args) > 1:
            end = args[1]
        deref(self.M).nsp.reset(new hif.PyNspFilter(start, end))

cdef class HIF_Mixed:
    """Python HIF object with single precision core

    The interfaces remain the same as the original user object, i.e.
    `hif::HIF<float,int>`. However, we significantly the parameters by
    hiding the needs of `hif::CRS`, `hif::CCS`, and `hif::Array`.
    Therefore, the interface is very generic and easily adapted to any other
    Python modules without the hassle of complete object types.
    
    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> A = random(10, 10, 0.5)
    >>> M = HIF_Mixed()
    >>> M.factorize(A)
    """
    cdef PyHIF_Mixed_ptr M
    cdef hif.Options params

    @staticmethod
    def is_mixed():
        """Indicated mixed precision"""
        return True

    def __init__(self):
        # for docstring purpose
        pass

    def __cinit__(self):
        self.M.reset(new hif.PyHIF_Mixed())
        self.params = hif.get_default_options()

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
        """Get the statistics information

        Parameters
        ----------
        entry: int
            entry field
        """
        return deref(self.M).stats(entry)

    def clear(self):
        """Clear internal memory usage"""
        deref(self.M).clear()

    def factorize(self, *args, shape=None, Params params=None):
        """Compute/build the preconditioner

        Parameters
        ----------
        *args : positional arguments
            either three array of CRS or scipy sparse matrix
        shape : tuple, optional
            if input is three arrays of CRS, then this must be given
        params : Params, optional
            control parameters, if ``None``, then use the default values

        See Also
        --------
        solve : solve for inv(HIF)*x
        """
        cdef:
            size_t n
            size_t m
        rowptr, colind, vals = convert_to_crs(*args, shape=shape)
        assert len(rowptr), "cannot deal with empty matrix"
        if params is not None:
            self.params = params.params
        n = len(rowptr) - 1
        m = 0
        factorize_M[PyHIF_Mixed_ptr](
            self.M, n, rowptr, colind, vals, m, &self.params
        )

    def solve(self, b, x=None, trans=False, r=0):
        r"""Core routine to use the preconditioner

        Essentially, this routine is to perform
        :math:`\boldsymbol{x}=\boldsymbol{M}^{-1}\boldsymbol{b}`, where
        :math:`\boldsymbol{M}` is our MILU preconditioner.

        Parameters
        ----------
        b : array_like
            right-hand side parameters
        x : array_like, optional
            solution vector, output buffer
        trans : bool, optional
            transpose/Hermitian flag, default is False
        r : int, optional
            final Schur complement rank

        Returns
        -------
        np.ndarray
            Solution vector
        """
        cdef size_t n
        bb = _as_value_array(b)
        assert len(bb.shape) == 1
        if x is None:
            xx = np.empty_like(bb)
        else:
            xx = _as_value_array(x)
        assert xx.shape == bb.shape, "inconsistent x and b"
        n = bb.size
        solve_M[PyHIF_Mixed_ptr](self.M, n, bb, xx, trans, r)
        return xx

    def hifir(
        self,
        *args,
        shape=None,
        N=2,
        x=None,
        trans=False,
        r=-1,
    ):
        """Access the preconditioner with iterative refinement

        Parameters
        ----------
        *args : positional arguments
            either three array of CRS or scipy sparse matrix, and rhs b
        shape : tuple, optional
            if input is three array, then this must be given
        N : int, optional
            explicit iteration steps, default is 2
        x : np.ndarray, optional
            for efficiency purpose, one can provide the buffer
        trans : bool, optional
            transpose/Hermitian flag, default is False
        r : int, optional
            final Schur complement rank

        Returns
        -------
        np.ndarray
            Solution vector
        """
        cdef:
            size_t n
            size_t NN
        rowptr, colind, vals, b = convert_to_crs_and_b(*args, shape=shape)
        if x is None:
            xx = np.empty_like(b)
        else:
            xx = _as_value_array(x)
        if xx.shape != b.shape:
            raise ValueError('inconsistent x and b')
        if N <= 0:
            N = 1
        n = b.size
        NN = N
        solve_M_IR[PyHIF_Mixed_ptr](
            self.M,
            n,
            rowptr,
            colind,
            vals,
            b,
            NN,
            xx,
            trans,
            r,
        )
        return xx

    def mmultiply(self, b, x=None, trans=False, r=0):
        r"""Core routine to use the preconditioner for matrix-vector

        Essentially, this routine is to perform
        :math:`\boldsymbol{x}=\boldsymbol{M}\boldsymbol{b}`, where
        :math:`\boldsymbol{M}` is our MILU preconditioner.

        Parameters
        ----------
        b : array_like
            right-hand side parameters
        x : array_like, optional
            solution vector, output buffer
        trans : bool, optional
            transpose/Hermitian flag, default is False
        r : int, optional
            final Schur complement rank

        Returns
        -------
        np.ndarray
            Solution vector
        """
        cdef size_t n
        bb = _as_value_array(b)
        assert len(bb.shape) == 1
        if x is None:
            xx = np.empty_like(bb)
        else:
            xx = _as_value_array(x)
        assert xx.shape == bb.shape, "inconsistent x and b"
        n = bb.size
        mmultiply_M[PyHIF_Mixed_ptr](self.M, n, bb, xx, trans, r)
        return xx

    def set_nsp_filter(self, *args):
        """Enable (right) null space filter

        One of the nice features in HIFIR (including HIFIR4PY) is to
        enabling (right) nullspace filter (eliminator) in preconditioner to
        solve singular systems.

        Parameters
        ----------
        arg1 : {None, int, callback}, optional
            First argument (later)
        arg2 : int, optional
            Second argument (later)

        Notes
        -----
        This function have the following usages:

        1. Set up a commonly used constant mode filter,
        2. Set up a constant mode filter within a certain range,
        3. Set up a filter via a user callback, and
        4. Unset the filter.

        For 1), this function takes no input argument. For 2), one or two
        arguments of integer can be passed, where the first one defines the
        starting index of the constant mode, whereas the pass-of-end position
        for the second one. For 3), the user needs to define a callback with
        a single input and output argument, where the input is guaranteed to
        be a 1D numpy ndarray, upon output, the user need to filter the null
        space components inplace to that vector. For 4), simply supplying
        ``None`` will remove the null space filter.

        Examples
        --------
        Given a preconditioner

        >>> M = create_M(mixed=True)

        We first consider the usage of 1)

        >>> M.set_nsp_filter()

        For 2)

        >>> M.set_nsp_filter(0)  # equiv as above
        >>> M.set_nsp_filter(0, 2)  # constant modes exist in [0, 2)

        For 3)

        >>> def my_filter(v):
        ...     v[:] -= np.sum(v) / v.size  # equiv as filter the constant mode
        ...
        >>> M.set_nsp_filter(my_filter)

        For 4)

        >>> M.set_nsp_filter(None)  # remove nullspace filter.
        """
        cdef:
            hif.PyNspFilter *py_nsp  # python null space filter
            size_t start
            size_t end
        if len(args) == 0:
            # simple constant mode
            deref(self.M).nsp.reset(new hif.PyNspFilter())
            return
        if args[0] is None:
            deref(self.M).nsp.reset()
            return
        if callable(args[0]):
            deref(self.M).nsp.reset(new hif.PyNspFilter())
            py_nsp = <hif.PyNspFilter *>deref(self.M).nsp.get()
            py_nsp.array_encoder = &raw2ndarray
            py_nsp.nsp_invoker = &call_user
            py_nsp.user_call = <PyObject*>args[0]
            Py_XINCREF(py_nsp.user_call)  # increment the reference
            py_nsp.enable_or()
            return
        start = args[0]
        end = <size_t>-1
        if len(args) > 1:
            end = args[1]
        deref(self.M).nsp.reset(new hif.PyNspFilter(start, end))


class KSP_Error(RuntimeError):
    """Base class of KSP error exceptions"""
    pass


class KSP_InvalidArgumentsError(KSP_Error):
    """Invalid input argument error"""
    pass


class KSP_MSolveError(KSP_Error):
    """preconditioner solve error"""
    pass


class KSP_DivergedError(KSP_Error):
    """Diverged error, i.e. maximum iterations exceeded"""
    pass


class KSP_StagnatedError(KSP_Error):
    """Stagnated error, i.e. no improvement amount two iterations in a row"""
    pass


class KSP_BreakDownError(KSP_Error):
    """Solver breaks down error"""
    pass


def _handle_flag(int flag):
    """handle KSP returned flag value"""
    if flag != hif.SUCCESS:
        if flag == hif.INVALID_ARGS:
            raise KSP_InvalidArgumentsError
        if flag == hif.M_SOLVE_ERROR:
            raise KSP_MSolveError
        if flag == hif.DIVERGED:
            raise KSP_DivergedError
        if flag == hif.STAGNATED:
            raise KSP_StagnatedError
        raise KSP_BreakDownError


def _handle_kernel(str kernel):
    cdef int kn
    if kernel == "tradition":
        kn = hif.TRADITION
    elif kernel == "iter-refine":
        kn = hif.ITERATIVE_REFINE
    elif kernel == "chebyshev-ir":
        kn = hif.CHEBYSHEV_ITERATIVE_REFINE
    else:
        choices = ("tradition", "iter-refine", "chebyshev-ir")
        raise KSP_InvalidArgumentsError(
            'invalid kernel {}, must be {}'.format(kernel, choices))
    return kn


cdef class KspSolver:
    r"""Flexible KSP base implementation with rhs preconditioner

    The KSP base implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.
    """
    cdef KSP_ptr solver

    @classmethod
    def is_mixed(cls):
        """Indicated if or not the solver is mixed precision enabled"""
        name = cls.__name__.split("_")
        if len(name) > 1:
            return name[1] == "Mixed"
        return False

    def __init__(self):
        pass

    def __cinit__(self):
        self.solver.reset()

    @property
    def rtol(self):
        "float: relative convergence tolerance (1e-6)"
        return deref(self.solver).get_rtol()

    @rtol.setter
    def rtol(self, double v):
        if v <= 0.0:
            raise ValueError('rtol must be positive')
        deref(self.solver).set_rtol(v)

    @property
    def maxit(self):
        """int: maximum number of interations (500)"""
        return deref(self.solver).get_maxit()

    @maxit.setter
    def maxit(self, int max_iters):
        if max_iters <= 0:
            raise ValueError('maxit must be positive integer')
        deref(self.solver).set_maxit(max_iters)

    @property
    def inner_steps(self):
        """int: inner iterations for IR-like inner iterations (4)"""
        return deref(self.solver).get_inner_steps()

    @inner_steps.setter
    def inner_steps(self, int maxi):
        if maxi <= 0:
            raise ValueError('inner_steps must be positive integer')
        deref(self.solver).set_inner_steps(maxi)

    @property
    def lamb1(self):
        """float: largest eigenvalue estimation"""
        return deref(self.solver).get_lamb1()

    @lamb1.setter
    def lamb1(self, double v):
        deref(self.solver).set_lamb1(v)

    @property
    def lamb2(self):
        """float: smallest eigenvalue estimation"""
        return deref(self.solver).get_lamb2()

    @lamb2.setter
    def lamb2(self, double v):
        deref(self.solver).set_lamb2(v)

    @property
    def resids(self):
        """list: list of history residuals"""
        cdef vector[double] res = \
            vector[double](deref(self.solver).get_resids_length())
        deref(self.solver).get_resids(res.data())
        return res

    def solve(
        self,
        *args,
        shape=None,
        x=None,
        kernel="tradition",
        init_guess=False,
        verbose=True,
        throw_on_error=True,
    ):
        """Sovle the rhs solution

        Parameters
        ----------
        *args : positional arguments
            either three array of CRS or scipy sparse matrix, and rhs b
        shape : tuple, optional
            if input is three array, then this must be given
        x : np.ndarray, optional
            for efficiency purpose, one can provide the buffer
        kernel : {"tradition","iter-refine","chebyshev-ir"}, optional
            kernel for preconditioning, default is traditional
        init_guess : bool, optional
            if ``False`` (default), then set initial state to be zeros
        verbose : bool, optional
            if ``True`` (default), then enable verbose printing
        throw_on_error: bool, optional
            if ``True`` (default), then raise exceptions

        Returns
        -------
        x : np.ndarray
            Solution array
        iters : int
            Number of iterations used
        flag : int
            Flag, if `throw_on_error` is ``False``, then this will be returned

        Raises
        ------
        KSP_InvalidArgumentsError
            invalid input arguments
        KSP_MSolveError
            preconditioner solver error, see :func:`HIF.solve`
        KSP_DivergedError
            iterations diverge due to exceeding :attr:`maxit`
        KSP_StagnatedError
            iterations stagnate
        KSP_BreakDownError
            solver breaks down
        """
        if init_guess and x is None:
            raise KSP_InvalidArgumentsError('init-guess missing x0')
        cdef:
            int kn = _handle_kernel(kernel)
            pair[int, size_t] info
            size_t n
        rowptr, colind, vals, b = convert_to_crs_and_b(*args, shape=shape)
        if x is None:
            xx = np.empty_like(b)
        else:
            xx = _as_value_array(x)
        if xx.shape != b.shape:
            raise ValueError('inconsistent x and b')
        n = b.size
        info = solve_KSP(
            self.solver,
            n,
            rowptr,
            colind,
            vals,
            b,
            xx,
            kn,
            init_guess,
            verbose,
        )
        if throw_on_error:
            _handle_flag(info.first)
            return xx, info.second
        return xx, info.second, info.first

    def __str__(self):
        fmt = self.__class__.__name__
        fmt += "\nrtol={}\nmaxit={}\n".format(self.rtol, self.maxit)
        return fmt

    def __repr__(self):
        return self.__str__()


cdef class GMRES(KspSolver):
    r"""GMRES implementation with rhs preconditioner

    The GMRES implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    restart : int, optional
        restart in GMRES, default is 30
    inner_steps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF()
    >>> M.factorize(A)
    >>> solver = GMRES(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(
        self,
        M=None,
        rtol=1e-6,
        restart=30,
        maxit=500,
        inner_steps=2,
        **kw
    ):
        pass

    def __cinit__(
        self,
        HIF M=None,
        double rtol=1e-6,
        int restart=30,
        int maxit=500,
        int inner_steps=2,
        **kw
    ):
        self.solver.reset(new hif.PyGMRES())
        if M is not None:
            deref(<gmres_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_restart(restart)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(inner_steps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop("lamb1", None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop("lamb2", None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def restart(self):
        """int: restart for GMRES (30)"""
        return deref(self.solver).get_restart()

    @restart.setter
    def restart(self, int rs):
        if rs <= 0:
            raise ValueError('restart must be positive integer')
        deref(self.solver).set_restart(rs)

    @property
    def M(self):
        """HIF: get preconditioner"""
        cdef:
            HIF _M = HIF()
            gmres_ptr child = <gmres_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF M):
        deref(<gmres_ptr>self.solver.get()).set_M(M.M)

    def __str__(self):
        fmt = self.__class__.__name__
        fmt += "\nrtol={}\nmaxit={}\nrestart={}\n".format(
            self.rtol, self.maxit, self.restart
        )
        return fmt


cdef class GMRES_Mixed(KspSolver):
    r"""Flexible GMRES implementation with rhs preconditioner (mixed precision)

    The GMRES implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF_Mixed, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    restart : int, optional
        restart in GMRES, default is 30
    inner_steps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF_Mixed()
    >>> M.factorize(A)
    >>> solver = GMRES_Mixed(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(
        self,
        M=None,
        rtol=1e-6,
        restart=30,
        maxit=500,
        inner_steps=2,
        **kw
    ):
        pass

    def __cinit__(
        self,
        HIF_Mixed M=None,
        double rtol=1e-6,
        int restart=30,
        int maxit=500,
        int inner_steps=2,
        **kw
    ):
        self.solver.reset(new hif.PyGMRES_Mixed())
        if M is not None:
            deref(<gmres_mixed_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_restart(restart)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(inner_steps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop("lamb1", None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop("lamb2", None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def restart(self):
        """int: restart for GMRES (30)"""
        return deref(self.solver).get_restart()

    @restart.setter
    def restart(self, int rs):
        if rs <= 0:
            raise ValueError('restart must be positive integer')
        deref(self.solver).set_restart(rs)

    @property
    def M(self):
        """HIF_Mixed: get preconditioner"""
        cdef:
            HIF_Mixed _M = HIF_Mixed()
            gmres_mixed_ptr child = <gmres_mixed_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF_Mixed M):
        deref(<gmres_mixed_ptr>self.solver.get()).set_M(M.M)

    def __str__(self):
        fmt = self.__class__.__name__
        fmt += "\nrtol={}\nmaxit={}\nrestart={}\n".format(
            self.rtol, self.maxit, self.restart
        )
        return fmt


cdef class FGMRES(KspSolver):
    r"""Flexible GMRES (FGMRES) implementation with rhs preconditioner

    The FGMRES implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    restart : int, optional
        restart in GMRES, default is 30
    inner_steps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF()
    >>> M.factorize(A)
    >>> solver = FGMRES(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(
        self,
        M=None,
        rtol=1e-6,
        restart=30,
        maxit=500,
        inner_steps=2,
        **kw
    ):
        pass

    def __cinit__(
        self,
        HIF M=None,
        double rtol=1e-6,
        int restart=30,
        int maxit=500,
        int inner_steps=2,
        **kw
    ):
        self.solver.reset(new hif.PyFGMRES())
        if M is not None:
            deref(<fgmres_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_restart(restart)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(inner_steps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop("lamb1", None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop("lamb2", None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def restart(self):
        """int: restart for GMRES (30)"""
        return deref(self.solver).get_restart()

    @restart.setter
    def restart(self, int rs):
        if rs <= 0:
            raise ValueError('restart must be positive integer')
        deref(self.solver).set_restart(rs)

    @property
    def M(self):
        """HIF: get preconditioner"""
        cdef:
            HIF _M = HIF()
            fgmres_ptr child = <fgmres_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF M):
        deref(<fgmres_ptr>self.solver.get()).set_M(M.M)

    def __str__(self):
        fmt = self.__class__.__name__
        fmt += "\nrtol={}\nmaxit={}\nrestart={}\n".format(
            self.rtol, self.maxit, self.restart
        )
        return fmt


cdef class FGMRES_Mixed(KspSolver):
    r"""Flexible GMRES implementation with rhs preconditioner (mixed precision)

    The FGMRES implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF_Mixed, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    restart : int, optional
        restart in GMRES, default is 30
    inner_steps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF_Mixed()
    >>> M.factorize(A)
    >>> solver = GMRES_Mixed(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(
        self,
        M=None,
        rtol=1e-6,
        restart=30,
        maxit=500,
        inner_steps=2,
        **kw
    ):
        pass

    def __cinit__(
        self,
        HIF_Mixed M=None,
        double rtol=1e-6,
        int restart=30,
        int maxit=500,
        int inner_steps=2,
        **kw
    ):
        self.solver.reset(new hif.PyFGMRES_Mixed())
        if M is not None:
            deref(<fgmres_mixed_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_restart(restart)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(inner_steps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop("lamb1", None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop("lamb2", None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def restart(self):
        """int: restart for GMRES (30)"""
        return deref(self.solver).get_restart()

    @restart.setter
    def restart(self, int rs):
        if rs <= 0:
            raise ValueError('restart must be positive integer')
        deref(self.solver).set_restart(rs)

    @property
    def M(self):
        """HIF_Mixed: get preconditioner"""
        cdef:
            HIF_Mixed _M = HIF_Mixed()
            fgmres_mixed_ptr child = <fgmres_mixed_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF_Mixed M):
        deref(<fgmres_mixed_ptr>self.solver.get()).set_M(M.M)

    def __str__(self):
        fmt = self.__class__.__name__
        fmt += "\nrtol={}\nmaxit={}\nrestart={}\n".format(
            self.rtol, self.maxit, self.restart
        )
        return fmt


cdef class FQMRCGSTAB(KspSolver):
    r"""Flexible QMRCGSTAB implementation with rhs preconditioner

    The FQMRCGSTAB implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    restart : int, optional
        restart in GMRES, default is 30
    innersteps : int, optional
        inner iterations used in IR style kernel, default is 4
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF()
    >>> M.factorize(A)
    >>> solver = FQMRCGSTAB(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(self, M=None, rtol=1e-6, maxit=500, innersteps=2, **kw):
        pass

    def __cinit__(
        self,
        HIF M=None,
        double rtol=1e-6,
        int maxit=500,
        int innersteps=2,
        **kw
    ):
        self.solver.reset(new hif.PyFQMRCGSTAB())
        if M is not None:
            deref(<fqmrcgstab_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(innersteps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop("lamb1", None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop("lamb2", None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def M(self):
        """HIF: get preconditioner"""
        cdef:
            HIF _M = HIF()
            fqmrcgstab_ptr child = <fqmrcgstab_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF M):
        deref(<fqmrcgstab_ptr>self.solver.get()).set_M(M.M)


cdef class FQMRCGSTAB_Mixed(KspSolver):
    r"""Flexible QMRCGSTAB implementation with rhs preconditioner (mixed-prec)

    The FQMRCGSTAB implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF_Mixed, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    innersteps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF_Mixed()
    >>> M.factorize(A)
    >>> solver = FQMRCGSTAB_Mixed(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(self, M=None, rtol=1e-6, maxit=500, innersteps=2, **kw):
        pass

    def __cinit__(
        self,
        HIF_Mixed M=None,
        double rtol=1e-6,
        int maxit=500,
        int innersteps=2,
        **kw,
    ):
        self.solver.reset(new hif.PyFQMRCGSTAB_Mixed())
        if M is not None:
            deref(<fqmrcgstab_mixed_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(innersteps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop('lamb1', None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop('lamb2', None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def M(self):
        """HIF_Mixed: get preconditioner"""
        cdef:
            HIF_Mixed _M = HIF_Mixed()
            fqmrcgstab_mixed_ptr child = <fqmrcgstab_mixed_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF_Mixed M):
        deref(<fqmrcgstab_mixed_ptr>self.solver.get()).set_M(M.M)


cdef class FBICGSTAB(KspSolver):
    r"""Flexible BICGSTAB implementation with rhs preconditioner

    The FBICGSTAB implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    restart : int, optional
        restart in GMRES, default is 30
    innersteps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF()
    >>> M.factorize(A)
    >>> solver = FBICGSTAB(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(self, M=None, rtol=1e-6, maxit=500, innersteps=2, **kw):
        pass

    def __cinit__(
        self,
        HIF M=None,
        double rtol=1e-6,
        int maxit=500,
        int innersteps=2,
        **kw,
    ):
        self.solver.reset(new hif.PyFBICGSTAB())
        if M is not None:
            deref(<fbicgstab_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(innersteps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop('lamb1', None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop('lamb2', None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def M(self):
        """HIF: get preconditioner"""
        cdef:
            HIF _M = HIF()
            fbicgstab_ptr child = <fbicgstab_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF M):
        deref(<fbicgstab_ptr>self.solver.get()).set_M(M.M)


cdef class FBICGSTAB_Mixed(KspSolver):
    r"""Flexible BICGSTAB implementation with rhs preconditioner (mixed-prec)

    The FBICGSTAB implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF_Mixed, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    innersteps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF_Mixed()
    >>> M.factorize(A)
    >>> solver = FBICGSTAB_Mixed(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(self, M=None, rtol=1e-6, maxit=500, innersteps=2, **kw):
        pass

    def __cinit__(
        self,
        HIF_Mixed M=None,
        double rtol=1e-6,
        int maxit=500,
        int innersteps=2,
        **kw,
    ):
        self.solver.reset(new hif.PyFBICGSTAB_Mixed())
        if M is not None:
            deref(<fbicgstab_mixed_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(innersteps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop('lamb1', None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop('lamb2', None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def M(self):
        """HIF_Mixed: get preconditioner"""
        cdef:
            HIF_Mixed _M = HIF_Mixed()
            fbicgstab_mixed_ptr child = <fbicgstab_mixed_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF_Mixed M):
        deref(<fbicgstab_mixed_ptr>self.solver.get()).set_M(M.M)


cdef class TGMRESR(KspSolver):
    r"""Truncated GMRESR implementation with rhs preconditioner

    The TGMRESR implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    cycle : int, optional
        truncated cycle in GMRESR, default is 10
    inner_steps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF()
    >>> M.factorize(A)
    >>> solver = TGMRESR(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(
        self,
        M=None,
        rtol=1e-6,
        cycle=10,
        maxit=500,
        inner_steps=2,
        **kw
    ):
        pass

    def __cinit__(
        self,
        HIF M=None,
        double rtol=1e-6,
        int cycle=10,
        int maxit=500,
        int inner_steps=2,
        **kw
    ):
        self.solver.reset(new hif.PyGMRES())
        if M is not None:
            deref(<fgmres_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_restart(cycle)  # use restart for cycle
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(inner_steps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop('lamb1', None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop('lamb2', None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def cycle(self):
        """int: cycle for TGMRESR (10)"""
        return deref(self.solver).get_restart()

    @cycle.setter
    def cycle(self, int cc):
        if cc <= 0:
            raise ValueError('cycle must be positive integer')
        deref(self.solver).set_restart(cc)

    @property
    def M(self):
        """HIF: get preconditioner"""
        cdef:
            HIF _M = HIF()
            fgmres_ptr child = <fgmres_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF M):
        deref(<fgmres_ptr>self.solver.get()).set_M(M.M)

    def __str__(self):
        fmt = self.__class__.__name__
        fmt += "\nrtol={}\nmaxit={}\ncycle={}\n".format(
            self.rtol, self.maxit, self.cycle
        )
        return fmt


cdef class TGMRESR_Mixed(KspSolver):
    r"""Truncated GMRESR implementation with rhs preconditioner (mixed precision)

    The TGMRESR implementation has two modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``iter-refine`` fashion, where
    :math:`\boldsymbol{M}` is treated as the splitted term in stationary
    iteration combining with the input matrix :math:`\boldsymbol{A}`.

    Parameters
    ----------
    M : HIF_Mixed, optional
        preconditioner
    rtol : float, optional
        relative tolerance, default is 1e-6
    maxit : int, optional
        maximum iterations, default is 500
    cycle : int, optional
        truncated cycle in GMRESR, default is 10
    inner_steps : int, optional
        inner iterations used in IR style kernel, default is 2
    lamb1 : float, optional
        if given, then used as the largest eigenvalue estimation
    lamb2 : float, optional
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hifir4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HIF_Mixed()
    >>> M.factorize(A)
    >>> solver = TGMRESR_Mixed(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """

    def __init__(self, M=None, rtol=1e-6, cycle=10, maxit=500, inner_steps=2, **kw):
        pass

    def __cinit__(
        self,
        HIF_Mixed M=None,
        double rtol=1e-6,
        int cycle=10,
        int maxit=500,
        int inner_steps=2,
        **kw,
    ):
        self.solver.reset(new hif.PyGMRES_Mixed())
        if M is not None:
            deref(<fgmres_mixed_ptr>self.solver.get()).set_M(M.M)
        deref(self.solver).set_rtol(rtol)
        deref(self.solver).set_restart(cycle)
        deref(self.solver).set_maxit(maxit)
        deref(self.solver).set_inner_steps(inner_steps)
        deref(self.solver).check_pars()
        lamb1 = kw.pop('lamb1', None)
        if lamb1 is not None:
            deref(self.solver).set_lamb1(lamb1)
        lamb2 = kw.pop('lamb2', None)
        if lamb2 is not None:
            deref(self.solver).set_lamb2(lamb2)

    @property
    def cycle(self):
        """int: cycle for TGMRESR (10)"""
        return deref(self.solver).get_restart()

    @cycle.setter
    def cycle(self, int cc):
        if cc <= 0:
            raise ValueError('cycle must be positive integer')
        deref(self.solver).set_restart(cc)

    @property
    def M(self):
        """HIF_Mixed: get preconditioner"""
        cdef:
            HIF_Mixed _M = HIF_Mixed()
            fgmres_mixed_ptr child = <fgmres_mixed_ptr>self.solver.get()
        if not deref(child).get_M():
            # empty
            deref(child).set_M(_M.M)
        else:
            _M.M = deref(child).get_M()
        return _M

    @M.setter
    def M(self, HIF_Mixed M):
        deref(<fgmres_mixed_ptr>self.solver.get()).set_M(M.M)

    def __str__(self):
        fmt = self.__class__.__name__
        fmt += "\nrtol={}\nmaxit={}\nrestart={}\n".format(
            self.rtol, self.maxit, self.restart
        )
        return fmt
