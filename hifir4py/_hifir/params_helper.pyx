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

# Parameter helpers
# Author(s):
#   Qiao Chen

from libcpp.string cimport string as std_string
from libcpp.vector cimport vector
cimport hifir4py


cdef extern from "hifir4py.hpp" namespace "hifir4py" nogil:
    vector[std_string] params_pos2tag


# Total number of parameters
__NUM_PARAMS__ = hifir4py.NUM_PARAMS

# verbose levels
__VERBOSE_NONE__ = hifir4py.VERBOSE_NONE
__VERBOSE_INFO__ = hifir4py.VERBOSE_INFO
__VERBOSE_PRE__ = hifir4py.VERBOSE_PRE
__VERBOSE_FAC__ = hifir4py.VERBOSE_FAC
__VERBOSE_PRE_TIME__ = hifir4py.VERBOSE_PRE_TIME

# reorderingoptions
__REORDER_OFF__ = hifir4py.REORDER_OFF
__REORDER_AUTO__ = hifir4py.REORDER_AUTO
__REORDER_AMD__ = hifir4py.REORDER_AMD
__REORDER_RCM__ = hifir4py.REORDER_RCM

# pivoting strategy
__PIVOTING_OFF__ = hifir4py.PIVOTING_OFF
__PIVOTING_ON__ = hifir4py.PIVOTING_ON
__PIVOTING_AUTO__ = hifir4py.PIVOTING_AUTO


def _tag2pos_helper():
    """A helper to construct a dict that maps parameter names to position"""
    cdef int i = 0
    t2p = {}
    for i in range(hifir4py.NUM_PARAMS):
        t2p[params_pos2tag[i].decode("utf-8")] = i
    return t2p


__PARAMS_TAG2POS__ = _tag2pos_helper()
