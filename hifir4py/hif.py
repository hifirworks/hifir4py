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
"""Main HIF module for ``hifir4py``

This module contains the core of HIFIR4PY, which includes the following twp
aspects:

1. control parameters, and
2. the HIF preconditioner.

.. module:: hifir4py.hif
    :noindex:
.. moduleauthor:: Qiao Chen <qiao.chen@stonybrook.edu>
"""

import enum
import numpy as np
from ._hifir import params_helper

__all__ = ["Params", "Verbose", "Reorder", "Pivoting", "HIF"]


class Verbose(enum.IntEnum):
    """Verbose level

    .. note:: All attributes are bit masks that support bit-wise ``or``
    """

    NONE = params_helper.__VERBOSE_NONE__
    """NONE mask, use to disable verbose
    """

    INFO = params_helper.__VERBOSE_INFO__
    """General info mask (set by default)
    """

    PRE = params_helper.__VERBOSE_PRE__
    """Enable verbose information with regards to preprocessing
    """

    FAC = params_helper.__VERBOSE_FAC__
    """Enable verbose for factorization

    .. warning:: This will slow down the factorization significantly!
    """

    PRE_TIME = params_helper.__VERBOSE_PRE_TIME__
    """Enable timing on preprocessing
    """


class Reorder(enum.IntEnum):
    """Reorder options
    """

    OFF = params_helper.__REORDER_OFF__
    """Disable reorder

    .. warning:: Not recommended!
    """

    AUTO = params_helper.__REORDER_AUTO__
    """Automatically determined reordering scheme (default)
    """

    AMD = params_helper.__REORDER_AMD__
    """Using approximate minimal degree (AMD) for all levels
    """

    RCM = params_helper.__REORDER_RCM__
    """Using reverse Cuthill-Mckee (RCM) for all levels
    """


class Pivoting(enum.IntEnum):
    """Pivoting options
    """

    OFF = params_helper.__PIVOTING_OFF__
    """Disable reorder
    """

    ON = params_helper.__PIVOTING_ON__
    """Enable pivoting"""

    AUTO = params_helper.__PIVOTING_AUTO__
    """Automatically determined reordering scheme (default)
    """


class Params:
    """Python interface of control parameters

    By default, each control parameter object is initialized with default
    values in the paper. In addition, modifying the parameters can be achieved
    by using key-value pairs, i.e. `__setitem__`. The keys are the names of
    those defined in original C/C++ ``struct``. A complete list of parameters
    can be retrieved by using :func:`keys`

    Examples
    --------

    >>> from hifir4py import *
    >>> params = Params()  # default parameters
    >>> params["verbose"] = Verbose.INFO | Verbose.FAC
    >>> params.reset()  # reset to default parameters
    """

    def __init__(self, **kw):
        self._params = np.zeros(params_helper.__NUM_PARAMS__)
        params_helper.set_default_params(self._params)
        self.__idx = 0  # iterator
        for k, v in kw.items():
            try:
                self[k] = v
            except KeyError:
                continue

    def to_dict(self):
        """Convert to dictionary"""
        _par = {}
        for k, i in params_helper.__PARAMS_TAG2POS__.items():
            if params_helper.__PARAM_DTYPES__[i]:
                _par[k] = self._params[i]  # float
            else:
                _par[k] = int(self._params[i])  # int
        return _par

    def __repr__(self):
        return repr(self.to_dict())

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, param_name: str):
        """Retrieve the parameter value given by its name

        Parameters
        ----------
        param_name : str
            Parameter name

        Returns
        -------
        int or float
            Parameter value
        """
        try:
            pos = params_helper.__PARAMS_TAG2POS__[param_name]
        except KeyError:
            raise KeyError("Unknown parameter name {}".format(param_name))
        # Determine data type
        if params_helper.__PARAM_DTYPES__[pos]:
            return self._params[pos]
        return int(self._params[pos])

    def __setitem__(self, param_name: str, v):
        """Set a configuration with keyvalue pair

        Parameters
        ----------
        param_name : str
            Option name
        v : int or float
            Corresponding value

        Raises
        ------
        KeyError
            Raised as per unsupported options
        """
        try:
            pos = params_helper.__PARAMS_TAG2POS__[param_name]
        except KeyError:
            raise KeyError("Unknown parameter name {}".format(param_name))
        self._params[pos] = v

    def __iter__(self):
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= params_helper.__NUM_PARAMS__:
            raise StopIteration
        try:
            return self.__getitem__(params_helper.__PARAMS_POS2TAG__[self.__idx])
        finally:
            self.__idx += 1

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def reset(self):
        """This function will reset all parameters to their default values"""
        params_helper.set_default_params(self._params)

    def enable_verbose(self, flag: int):
        """Enable verbose level

        Parameters
        ----------
        flag : int
            Verbose level in :class:`Verbose`

        See Also
        --------
        disable_verbose: Disable verbose
        """
        if flag == Verbose.NONE:
            return
        cur_flag = self["verbose"]
        if cur_flag != Verbose.NONE:
            self["verbose"] |= flag
        else:
            self["verbose"] = flag

    def disable_verbose(self):
        """Disable verbose"""
        self["verbose"] = Verbose.NONE


def _create_cpphif(index_t, is_complex: bool, is_mixed: bool):
    """Select proper C++ preconditioner"""
    from . import _hifir

    assert index_t in (np.int32, np.int64), "Must be int32 or int64"
    cpphif = "{}i{}hif"
    if index_t == np.int32:
        index_size = 32
    else:
        index_size = 64
    if not is_mixed:
        vd = "d" if not is_complex else "z"
    else:
        vd = "s" if not is_complex else "c"
    return getattr(_hifir, cpphif.format(vd, index_size)).HIF()


def _tocsr(A):
    """Helper function to convert to CRS"""
    import scipy.sparse

    if not scipy.sparse.issparse(A):
        raise TypeError("Must be SciPy sparse matrix")
    if not scipy.sparse.isspmatrix_csr(A):
        return A.tocsr()
    return A


_tocrs = _tocsr


def _ensure_similar(A, S):
    """Helper to make sure two CRS matrices are similar"""
    assert A.indptr.dtype == S.indptr.dtype, "Unmatched index type"
    assert A.data.dtype == S.data.dtype, "Unmatched value type"
    assert A.shape == S.shape, "Unmatched sizes"


def _must_1d(x):
    """Helper function to ensure array must be 1D"""
    if (len(x.shape) > 1 and x.shape[1] != 1) or len(x.shape) > 2:
        raise ValueError("Must be 1D array")


class HIF:
    """The HIF preconditioner object

    This class implements the flexible interface of the HIF preconditioner in
    Python, with supports of both mixed-precision computation and complex
    arithmetic.

    Attributes
    ----------
    M_
    A
    S
    levels
    nnz
    nnz_ef
    nnz_ldu
    nrows
    ncols
    rank
    schur_rank
    schur_size

    Examples
    --------
    One can simply create an empty preconditioner, and initialize it later.

    >>> from hifir4py import *
    >>> hif = HIF()

    Alternatively, one can create an instance and factorize it.

    >>> from scipy.sparse import rank
    >>> A = rand(10, 10, 0.5)
    >>> hif = HIF(A)

    The latter usages shares the same interface as :meth:`~.HIF.factorize` does.
    """

    def __init__(self, A=None, **kw):
        self.__hif = None
        self._A = A
        self._S = kw.pop("S", None)
        # If both A and S are None, then we skip
        if self._A is None and self._S is None:
            return
        if self._A is None:
            self._A = self._S
        self.factorize(self._A, S=self._S, **kw)

    @property
    def M_(self):
        """Get the underlying C++ HIF object

        .. warning:: Access this only if you know what you are doing!
        """
        return self.__hif

    @property
    def A(self):
        """:class:`~scipy.sparse.csr_matrix`: CRS (CSR) matrix"""
        return self._A

    @property
    def S(self):
        """:class:`~scipy.sparse.csr_matrix`: CRS (CSR) sparsifier"""
        return self._S

    @property
    def levels(self):
        """int: Number of levels"""
        self._make_sure_not_null()
        return self.__hif.levels

    @property
    def nnz(self):
        """int: Number of nonzeros in the preconditioner"""
        self._make_sure_not_null()
        return self.__hif.nnz

    @property
    def nnz_ef(self):
        """int: Number of nonzeros in E and F pars"""
        self._make_sure_not_null()
        return self.__hif.nnz_ef

    @property
    def nnz_ldu(self):
        """int: Number of nonzeros in the L, D, and U factors"""
        self._make_sure_not_null()
        return self.__hif.nnz_ldu

    @property
    def nrows(self):
        """int: Number of rows"""
        self._make_sure_not_null()
        return self.__hif.nrows

    @property
    def ncols(self):
        """int: Number of columns"""
        self._make_sure_not_null()
        return self.__hif.ncols

    @property
    def rank(self):
        """int: Numerical rank"""
        self._make_sure_not_null()
        return self.__hif.rank

    @property
    def schur_rank(self):
        """int: Numerical rank of the final Schur complement"""
        self._make_sure_not_null()
        return self.__hif.schur_rank

    @property
    def schur_size(self):
        """int: Size of the final Schur complement"""
        self._make_sure_not_null()
        return self.__hif.schur_size

    def is_null(self):
        """Check if the underlying C++ HIF is ``None``"""
        return self.__hif is None

    def empty(self):
        """Check emptyness"""
        return self.is_null() or self.__hif.empty()

    def _make_sure_not_null(self):
        if self.is_null():
            raise AttributeError("The underlying C++ HIF attribute is missing")

    def is_mixed(self):
        """Check if the underlying HIF is mixed-precision

        Raises
        ------
        AttributeError
            This is raised if the underlying C++ HIF attribute is missing

        See Also
        --------
        is_complex
        """
        self._make_sure_not_null()
        return self.__hif.is_mixed()

    def is_complex(self):
        """Check if the underlying C++ HIF is complex

        Raises
        ------
        AttributeError
            This is raised if the underlying C++ HIF attribute is missing

        See Also
        --------
        is_mixed
        """
        self._make_sure_not_null()
        return self.__hif.is_complex()

    def index_size(self):
        """Check the integer byte size used in underlying C++ HIF

        Raises
        ------
        AttributeError
            This is raised if the underlying C++ HIF attribute is missing
        """
        self._make_sure_not_null()
        return self.__hif.index_size()

    def factorize(self, A, **kw):
        """Factorize a HIF preconditioner

        This function is the core in HIF to (re)factorize a HIF preconditioner
        given input a matrix or sparsifier.

        Parameters
        ----------
        A : :class:`~scipy.sparse.csr_matrix`
            Input CRS matrix
        S : :class:`~scipy.sparse.csr_matrix` or None, optional
            Optional sparsifier input (on which we will compute HIF)
        is_mixed : bool, optional
            Whether or not using mixed-precision (using single)
        params : :class:`.Params`, optional
            Control parameters, using default values if not provided

        Examples
        --------
        >>> from scipy.sparse import rand
        >>> from hifir4py import *
        >>> A = rand(10, 10, 0.5)
        >>> hif = HIF()
        >>> hif.factorize(A)

        Notes
        -----
        Besides passing in a :class:`.Params` object for control parameters,
        one can also directly using key-value pairs for parameters while calling
        the factorize function.

        See Also
        --------
        apply
        """
        self._A = _tocrs(A)
        self._S = kw.pop("S", self._A)
        if self._S is None:
            self._S = self._A
        self._S = _tocrs(self._S)
        _ensure_similar(self._A, self._S)
        is_complex = np.iscomplexobj(self._S)
        is_mixed = kw.pop("is_mixed", False)
        params = kw.pop("params", Params(**kw))
        assert isinstance(params, Params), "Parameters must be Params type"
        if (
            self.is_null()
            or self.index_size() != self._S.indptr.dtype.itemsize
            or self.is_mixed() != is_mixed
            or self.is_complex() != is_complex
        ):
            self.__hif = _create_cpphif(self._S.indptr.dtype, is_complex, is_mixed)
        self.__hif.factorize(
            self._S.indptr, self._S.indices, self._S.data, params._params
        )

    def apply(self, b, **kw):
        """Apply the preconditioner with a given operation (op)

        This function is to apply the preconditioner in the following four
        different modes:

        1. multilevel triangular solve ("S"),
        2. transpose/Hermitian multilevel triangular solve ("SH" or "ST"),
        3. multilevel matrix-vector multiplication ("M"), and
        4. tranpose/Hermitian multilevel matrix-vector multiplication ("MT" or
           "MH").

        Parameters
        ----------
        b : :class:`~numpy.ndarray`
            Input RHS vector
        op : str, optional
            Operation option, in {"S", "SH", "ST", "M", "MH", "MT"} and case
            insensitive; note that "SH" and "ST" are the same
        x : :class:`~numpy.ndarray`, optional
            Output result, can be passed in as workspace
        nirs : int, optional
            Number of iterative refinement steps (default 1)
        rank : int, optional
            Numerical rank used in final Schur complement

        Examples
        --------
        Given an instance of HIF, say ``hif``. The following computes the
        standard triangular solve in preconditioners

        >>> x = hif.apply(b)

        To enable transpose/Hermitian, one can do

        >>> x = hif.apply(b, op="ST")

        For triangular solve, the following enables two-step IR

        >>> x = hif.apply(b, nirs=2)

        Finally, the following demonstrates how to apply matrix-vector product
        operation

        >>> x = hif.apply(b, op="M")
        >>> x = hif.apply(b, op="MT")  # tranpose/Hermitian

        See Also
        --------
        factorize
        schur_rank
        """
        self._make_sure_not_null()
        if self.empty():
            raise ValueError("The preconditioner is still empty")
        op = kw.pop("op", "s")
        op = op.lower()
        assert op in ("s", "sh", "st", "m", "mh", "mt"), "Unknown operation {}".format(
            op
        )
        _must_1d(b)
        if self._S.shape[0] != b.shape[0]:
            raise ValueError("Unmatched sizes of input vector and preconditioner")
        nirs = kw.pop("nirs", 1)
        rank = kw.pop("rank", -1) if op[0] == "s" and nirs > 1 else kw.pop("rank", 0)
        trans = len(op) > 1
        # buffer
        x = kw.pop("x", np.empty(b.shape[0], dtype=b.dtype))
        _must_1d(x)
        assert x.shape[0] == b.shape[0], "Unmatched sizes for input buffer x"
        if op[0] == "m":
            self.__hif.mmultiply(b.reshape(-1), x.reshape(-1), trans, rank)
        else:
            if nirs <= 1:
                self.__hif.solve(b.reshape(-1), x.reshape(-1), trans, rank)
            else:
                self.__hif.hifir(
                    self._A.indptr,
                    self._A.indices,
                    self._A.data,
                    b.reshape(-1),
                    nirs,
                    x.reshape(-1),
                    trans,
                    rank,
                )
        return x

    def to_scipy(self):
        """Compute a SciPy LinearOperator based on HIF

        Returns
        -------
        :class:`~scipy.sparse.linalg.LinearOperator`
            Return a linear operator in SciPy so that HIF can be used in its
            built-in KSP solvers.
        """
        self._make_sure_not_null()
        if self.empty():
            raise ValueError("Preconditioner is still empty")

        import scipy.sparse.linalg as spla

        return spla.LinearOperator((self.nrows, self.ncols), lambda b: self.apply(b))
