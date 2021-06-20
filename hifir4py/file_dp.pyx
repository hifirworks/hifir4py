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

# This file is to utilize Cython API feature to generate Python3 print
# wrappers for stdout (1) and stderr (2), which will be wrapped as
# HIF_{STDOUT,STDERR}, resp.

# Authors:
#   Qiao,

import sys


cdef api void hifir4py_stdout(const char *msg):
    print(msg.decode("utf-8"), file=sys.stdout)


cdef api void hifir4py_stderr(const char *msg):
    print(msg.decode("utf-8"), file=sys.stderr)
