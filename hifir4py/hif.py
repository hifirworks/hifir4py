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
"""Main HIF module for ``hifir4py``"""
import enum
import typing
import numpy as np
import scipy.sparse.linalg as spla
from . import _hifir
from .utils import to_crs, must_1d, ensure_same, Tuple

__all__ = ["Params", "Verbose", "Reorder", "Pivoting", "HIF"]


class Verbose(enum.IntEnum):
    """Verbose level

    .. note:: All attributes are bit masks that support bit-wise ``or``
    """

    NONE = _hifir.params_helper.__VERBOSE_NONE__
    """NONE mask, use to disable verbose
    """

    INFO = _hifir.params_helper.__VERBOSE_INFO__
    """General info mask (set by default)
    """

    PRE = _hifir.params_helper.__VERBOSE_PRE__
    """Enable verbose information with regards to preprocessing
    """

    FAC = _hifir.params_helper.__VERBOSE_FAC__
    """Enable verbose for factorization

    .. warning:: This will slow down the factorization significantly!
    """

    PRE_TIME = _hifir.params_helper.__VERBOSE_PRE_TIME__
    """Enable timing on preprocessing
    """


class Reorder(enum.IntEnum):
    """Reorder options"""

    OFF = _hifir.params_helper.__REORDER_OFF__
    """Disable reorder

    .. warning:: Not recommended!
    """

    AUTO = _hifir.params_helper.__REORDER_AUTO__
    """Automatically determined reordering scheme (default)
    """

    AMD = _hifir.params_helper.__REORDER_AMD__
    """Using approximate minimal degree (AMD) for all levels
    """

    RCM = _hifir.params_helper.__REORDER_RCM__
    """Using reverse Cuthill-Mckee (RCM) for all levels
    """


class Pivoting(enum.IntEnum):
    """Pivoting options"""

    OFF = _hifir.params_helper.__PIVOTING_OFF__
    """Disable reorder
    """

    ON = _hifir.params_helper.__PIVOTING_ON__
    """Enable pivoting"""

    AUTO = _hifir.params_helper.__PIVOTING_AUTO__
    """Automatically determined reordering scheme (default)
    """


class Params:
    """Python interface of control parameters

    By default, each control parameter object is initialized with default
    values in the paper. In addition, modifying the parameters can be achieved
    by using key-value pairs, i.e. ``__setitem__``. The keys are the names of
    those defined in original C/C++ ``struct``. A complete list of parameters
    can be retrieved by using :func:`~.Params.keys`.

    Examples
    --------

    >>> from hifir4py import *
    >>> params = Params()  # default parameters
    >>> params["verbose"] = Verbose.INFO | Verbose.FAC
    >>> params.reset()  # reset to default parameters
    """

    def __init__(self, **kw):
        """Create a parameter object

        One can potentially pass key-value pairs to initialize a control
        parameter object.

        Examples
        --------
        The following creates a parameter with scalability-oriented dropping
        thresholds 3.

        >>> params = Params(alpha_L=3, alpha_U=3)
        """
        self._params = np.zeros(_hifir.params_helper.__NUM_PARAMS__)
        _hifir.params_helper.set_default_params(self._params)
        self.__idx = 0  # iterator
        for k, v in kw.items():
            try:
                self[k] = v
            except KeyError:
                continue

    @property
    def tau(self) -> float:
        """float: Drop tolerances for both L and U factors"""
        return self["tau_L"]

    @tau.setter
    def tau(self, v: float):
        self["tau_L"] = self["tau_U"] = v

    @property
    def kappa(self) -> float:
        """float: Conditioning thresholds for L, D, and U factors"""
        return self["kappa"]

    @kappa.setter
    def kappa(self, v: float):
        self["kappa"] = self["kappa_d"] = v

    @property
    def alpha(self) -> float:
        """float: Scalability-oriented dropping thresholds for both L and U factors"""
        return self["alpha_L"]

    @alpha.setter
    def alpha(self, v: float):
        self["alpha_L"] = self["alpha_U"] = v

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        _par = {}
        for k, i in _hifir.params_helper.__PARAMS_TAG2POS__.items():
            if _hifir.params_helper.__PARAM_DTYPES__[i]:
                _par[k] = self._params[i]  # float
            else:
                _par[k] = int(self._params[i])  # int
        return _par

    def __repr__(self) -> str:
        return repr(self.to_dict())

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, param_name: str) -> typing.Union[float, int]:
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
            pos = _hifir.params_helper.__PARAMS_TAG2POS__[param_name]
        except KeyError as e:
            raise KeyError("Unknown parameter name {}".format(param_name)) from e
        # Determine data type
        if _hifir.params_helper.__PARAM_DTYPES__[pos]:
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
            pos = _hifir.params_helper.__PARAMS_TAG2POS__[param_name]
        except KeyError as e:
            raise KeyError("Unknown parameter name {}".format(param_name)) from e
        self._params[pos] = v

    def __iter__(self) -> str:
        self.__idx = 0
        return self

    def __next__(self):
        if self.__idx >= _hifir.params_helper.__NUM_PARAMS__:
            raise StopIteration
        try:
            return self.__getitem__(_hifir.params_helper.__PARAMS_POS2TAG__[self.__idx])
        finally:
            self.__idx += 1

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def reset(self) -> None:
        """This function will reset all parameters to their default values"""
        _hifir.params_helper.set_default_params(self._params)

    def enable_verbose(self, flag: int) -> None:
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

    def disable_verbose(self) -> None:
        """Disable verbose"""
        self["verbose"] = Verbose.NONE


class HIF:
    """The HIF preconditioner object

    This class implements the flexible interface of the HIF preconditioner in
    Python, with supports of both mixed-precision computation and complex
    arithmetic.

    Examples
    --------
    One can simply create an empty preconditioner, and initialize it later.

    >>> from hifir4py import *
    >>> hif = HIF()

    Alternatively, one can create an instance and factorize it.

    >>> from scipy.sparse import rand
    >>> A = rand(10, 10, 0.5)
    >>> hif = HIF(A)

    The latter usages shares the same interface as :meth:`~.HIF.factorize` does.
    """

    def __init__(self, A=None, **kw):
        """Create a HIF preconditioner

        One can construct an empty preconditioner, i.e.,

        >>> M = HIF()

        Alternatively, one can construct and factorize at the same time, and
        see :func:`factorize` for the interface.
        """
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
        """Underlying C++ HIF object

        .. warning:: Access this only if you know what you are doing!
        """
        return self.__hif

    @property
    def A(self):
        """:class:`~scipy.sparse.csr_matrix`: User-input CRS (CSR) matrix"""
        return self._A

    @A.setter
    def A(self, A):
        self._A = to_crs(A)

    @property
    def S(self):
        """:class:`~scipy.sparse.csr_matrix`: User-input CRS (CSR) sparsifier"""
        return self._S

    @property
    def levels(self) -> int:
        """int: Number of levels"""
        return 0 if self.empty() else self.__hif.levels

    @property
    def nnz(self) -> int:
        """int: Total number of nonzeros in the preconditioner"""
        return 0 if self.empty() else self.__hif.nnz

    @property
    def nnz_ef(self) -> int:
        """int: Total number of nonzeros in the E and F off diagonal blocks"""
        return 0 if self.empty() else self.__hif.nnz_ef

    @property
    def nnz_ldu(self) -> int:
        """int: Total number of nonzeros in the L, D, and U factors"""
        return 0 if self.empty() else self.__hif.nnz_ldu

    @property
    def nrows(self) -> int:
        """int: Number of rows"""
        return 0 if self.empty() else self.__hif.nrows

    @property
    def ncols(self) -> int:
        """int: Number of columns"""
        return 0 if self.empty() else self.__hif.ncols

    @property
    def shape(self) -> Tuple[int, int]:  # pylint: disable=unsubscriptable-object
        """tuple: 2-tuple of the preconditioner shape, i.e., (:attr:`nrows`, :attr:`ncols`)"""
        return (self.nrows, self.ncols)

    @property
    def rank(self) -> int:
        """int: Numerical rank of the preconditioner"""
        return 0 if self.empty() else self.__hif.rank

    @property
    def schur_rank(self) -> int:
        """int: Numerical rank of the final Schur complement"""
        return 0 if self.empty() else self.__hif.schur_rank

    @property
    def schur_size(self) -> int:
        """int: Size of the final Schur complement"""
        return 0 if self.empty() else self.__hif.schur_size

    def empty(self) -> bool:
        """Check emptyness"""
        return self.__hif is None or self.__hif.empty()

    def update(self, A) -> None:
        """Update the A matrix used in IR

        .. note:: This function does not perform factorization on ``A``.
        """
        self.A = A

    def __repr__(self) -> str:
        """Representation of a HIF instance"""
        if self.empty():
            return "Empty HIF preconditioner"
        rp = """HIF preconditioner
        shape: {}
        nnz: {}
        levels: {}
        nnz-ratio: {:.2f}%
        """.format(
            self.shape, self.nnz, self.levels, 100.0 * self.nnz / self._S.nnz
        )
        return rp

    def __str__(self) -> str:
        return repr(self)

    def is_mixed(self) -> bool:
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

    def is_complex(self) -> bool:
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

    def index_size(self) -> int:
        """Check the integer byte size used in underlying C++ HIF

        Raises
        ------
        AttributeError
            This is raised if the underlying C++ HIF attribute is missing
        """
        self._make_sure_not_null()
        return self.__hif.index_size()

    def clear(self) -> None:
        """Clear the internal memory for the preconditioner"""
        self.__hif = None

    def factorize(self, A, **kw) -> None:
        """Factorize a HIF preconditioner

        This function is the core in HIF to (re)factorize a HIF preconditioner
        given input a matrix or sparsifier.

        Parameters
        ----------
        A : :class:`~scipy.sparse.csr_matrix`
            Input CRS matrix
        S : :class:`~scipy.sparse.csr_matrix` or ``None``, optional
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
        self._A = to_crs(A)
        self._S = kw.pop("S", self._A)
        if self._S is None:
            self._S = self._A
        self._S = to_crs(self._S)
        _ensure_similar(self._A, self._S)
        is_complex = np.iscomplexobj(self._S)
        is_mixed = kw.pop("is_mixed", False)
        params = kw.pop("params", Params(**kw))
        if not issubclass(params.__class__, Params):
            raise TypeError("Parameters must be Params type")
        if (
            self.__hif is None
            or self.index_size() != self._S.indptr.dtype.itemsize
            or self.is_mixed() != is_mixed
            or self.is_complex() != is_complex
        ):
            self.__hif = _create_cpphif(self._S.indptr.dtype, is_complex, is_mixed)
        self.__hif.factorize(self._S.indptr, self._S.indices, self._S.data, params._params)

    def refactorize(self, S, **kw) -> None:
        """Refactorize a sparsifier

        .. note::

            This function is similar to :func:`factorize`, but differs in that
            this function doesn't update the A matrix.

        Parameters
        ----------
        S : :class:`~scipy.sparse.csr_matrix`
            Sparsifier input (on which we will compute HIF)
        is_mixed : bool, optional
            Whether or not using mixed-precision (using single)
        params : :class:`.Params`, optional
            Control parameters, using default values if not provided

        See Also
        --------
        update

        Examples
        --------

        >>> from scipy.sparse import rand
        >>> from hifir4py import *
        >>> hif = HIF(rand(10, 10, 0.5))
        >>> A1 = hif.A
        >>> A2 = rand(10, 10, 0.5)
        >>> hif.refactorize(A2)
        >>> assert A1 is hif.A

        The following example illustrates how to update IR operator ``A`` and
        factorization asynchronously. In particular, we refactorize every 10
        iterations, but update IR operator for every step.

        >>> import numpy as np
        >>> from scipy.sparse import rand
        >>> from hifir4py import *
        >>> hif = HIF(rand(10, 10, 0.5)) # initial HIF
        >>> b = np.random.rand(10)
        >>> for i in range(100):
        >>>     x = hif.apply(b, nirs=2)  # 2-iter refinement with hif.A
        >>>     A = rand(10, 10, 0.5)
        >>>     hif.update(A)  # update A, or equiv as hif.A = A
        >>>     if i % 10 == 0:
        >>>         hif.refactorize(A)  # refactorization
        """
        A_bak = self._A
        kw.pop("S", None)  # remove S in kw
        self.factorize(S, **kw)
        if A_bak is not None:
            _ensure_similar(A_bak, self._S)
            self._A = A_bak  # resume to the old A

    def apply(self, b: np.ndarray, **kw) -> np.ndarray:
        r"""Apply the preconditioner with a given operation (op)

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
        betas : float or list, optional
            The residual bound for iterative refinement. If a scalar is passed
            in, then it is assumed to be lower bound (:math:`\beta_L`) and the
            upper bound (:math:`\beta_U`) is set to be maximum value.
        nirs : int, optional
            Number of iterative refinement steps (default 1)
        rank : int, optional
            Numerical rank used in final Schur complement

        Returns
        -------
        x : :class:`~numpy.ndarray`
            Computed solution vector
        iters : int, optional
            If iterative refinement is enabled in triangular solve and residual
            bounds betas is passed in, then this indicates the actual refinement
            iterations.
        flag : int, optional
            If iterative refinement is enabled in triangular solve and residual
            bounds betas is passed in, then this indicates the status of
            iterative refinement. If flag==0, then the IR process converged;
            if flag>0, then it diverged; otherwise, it reached maximum
            iterative refinement limit (nirs).

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
        op = kw.pop("op", "s").lower()
        if op not in ("s", "sh", "st", "m", "mh", "mt"):
            raise ValueError("Unknown operation {}".format(op))
        must_1d(b)
        ensure_same(self._S.shape[0], b.shape[0])
        nirs = kw.pop("nirs", 1)
        rank = (
            kw.pop("rank", np.iinfo("uint64").max)
            if op[0] == "s" and nirs > 1
            else kw.pop("rank", 0)
        )
        if rank < 0:
            rank = np.iinfo("uint64").max
        trans = len(op) > 1
        # buffer
        x = kw.pop("x", np.empty(b.shape[0], dtype=b.dtype))
        must_1d(x)
        ensure_same(x.shape[0], b.shape[0])
        flag = None
        if op[0] == "m":
            self.__hif.mmultiply(b.reshape(-1), x.reshape(-1), trans, rank)
        else:
            if nirs <= 1:
                self.__hif.solve(b.reshape(-1), x.reshape(-1), trans, rank)
            else:
                betas = kw.pop("betas", None)
                if betas is None:
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
                else:
                    if np.isscalar(betas):
                        betas = [float(betas), np.finfo("double").max]
                    betas = np.asarray(betas, dtype=float)
                    iters, flag = self.__hif.hifir_beta(
                        self._A.indptr,
                        self._A.indices,
                        self._A.data,
                        b.reshape(-1),
                        nirs,
                        betas.reshape(-1),
                        x.reshape(-1),
                        trans,
                        rank,
                    )
        if flag is None:
            return x
        return x, iters, flag

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

        return spla.LinearOperator((self.nrows, self.ncols), self.apply)

    def _make_sure_not_null(self) -> None:
        if self.__hif is None:
            raise AttributeError("The underlying C++ HIF attribute is missing")


def _create_cpphif(index_t: np.dtype, is_complex: bool, is_mixed: bool):
    """Select proper C++ preconditioner"""
    if index_t not in (np.int32, np.int64):
        raise ValueError("Must be int32 or int64")
    cpphif = "{}i{}hif"
    index_size = 32 if index_t == np.int32 else 64
    if not is_mixed:
        vd = "d" if not is_complex else "z"
    else:
        vd = "s" if not is_complex else "c"
    return getattr(_hifir, cpphif.format(vd, index_size)).HIF()


def _ensure_similar(A, S) -> None:
    """Helper to make sure two CRS matrices are similar"""
    ensure_same(A.indptr.dtype, S.indptr.dtype, "Unmatched index type")
    ensure_same(A.data.dtype, S.data.dtype, "Unmatched value type")
    ensure_same(A.shape, S.shape, "Unmatched shapes")
