Python3 Interface for HILUCSI
=============================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Welcome to the Python3 interface of HILUCSI package--- *hilucsi4py*. The Python
interface is implemented with Cython, and the Cython interface is also
available to use.

Dependencies
------------

*psmilu4py* requires *Cython* and *setuptools* during installation and
compilation. The must-have runtime package is *numpy*. It implicitly supports
*scipy* sparse matrices as well as the `LinearOperator` interface for using
as its built-in KSP's preconditioners.

Installation
-------------

You need to configure linking against LAPACK by setting the environment
variable ``HILUCSI_LAPACK_LIB``, and the default is ``-llapack``. If you
have a specific library path to LAPACK, you then need to set the environment
variable ``HILUCSI_LAPACK_LIB_PATH``.

To sum up, the following environment variables can be configured

1. ``HILUCSI_LAPACK_LIB``, default is ``-llapack``
2. ``HILUCSI_LAPACK_LIB_PATH``, default is empty

The default installation
````````````````````````

The following command assumes ``HILUCSI.hpp`` is located in system include
path or current directory. In addition, ``liblapack.so`` can be found in system
library path or under ``LIBRARY_PATH``.

.. code:: console

    pip3 install . --user

Installation with customized third-party libraries
``````````````````````````````````````````````````

Sometimes, it's helpful to have optimized LAPACK package. The following command
shows how to link MKL (on Ubuntu).

.. code:: console

    export HILUCSI_LAPACK_LIB="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core"
    export HILUCSI_LAPACK_LIB_PATH=/opt/intel/mkl/lib/intel64
    pip3 install . --user

Contacts
--------

Qiao (Chiao) Chen, <qiao.chen@stonybrook.edu>
