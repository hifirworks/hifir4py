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

__all__ = ["Params", "Verbose", "Reorder", "Pivoting"]


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
            self[k] = v

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
