Python3 Interface for HIFIR (``hifir4py``)
==========================================

.. image:: https://github.com/hifirworks/hifir4py/actions/workflows/python-app.yml/badge.svg?branch=main
    :target: https://github.com/hifirworks/hifir4py/actions/workflows/python-app.yml
.. image:: https://img.shields.io/pypi/v/hifir4py.svg?branch=main
    :target: https://pypi.org/project/hifir4py/
.. image:: https://img.shields.io/pypi/pyversions/hifir4py.svg?style=flat-square
    :target: https://pypi.org/project/hifir4py/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Welcome to the Python3 interface of HIFIR package--- *hifir4py*. The Python
interface is implemented with Cython.

Detailed documentation of ``hifir4py`` can be found at `<https://hifirworks.github.io/hifir4py>`_.

Installation
-------------

One can simply use ``pip`` to install ``hifir4py``, i.e.,

.. code:: console

    pip3 install hifir4py --user

For Windows users, we recommend using `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/>`_
or `Cygwin <https://www.cygwin.com/>`_.

Installation from Source
------------------------

It is also possible to install from the source. First, down load the most
recent stable release

.. code:: console

    git clone -b release https://github.com/hifirworks/hifir4py.git

You can update to the most recent version via ``git pull`` since ``git clone``
or last ``git pull``. In addition, you can download the archives at
`<https://github.com/hifirworks/hifir4py/releases>`_.

You need to configure linking against LAPACK by setting the environment
variable ``HIFIR_LAPACK_LIB`` whose default is ``-llapack``. If you
have a specific library path to LAPACK, you then need to set the environment
variable ``HIFIR_LAPACK_LIB_PATH``.

To sum up, the following environment variables can be configured

1. ``HIFIR_LAPACK_LIB``, default is ``-llapack``
2. ``HIFIR_LAPACK_LIB_PATH``, default is empty

.. code:: console

    pip3 install . --user

Installation with customized LAPACK/BLAS
````````````````````````````````````````

Sometimes, it's helpful to have optimized LAPACK package. The following command
shows how to link MKL (on Ubuntu).

.. code:: console

    HIFIR_LAPACK_LIB="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" HIFIR_LAPACK_LIB_PATH=/opt/intel/mkl/lib/intel64 pip3 install . --user

Copyrights & Licenses
---------------------

``hifir4py`` is developed and maintained by the NumGeom Research Group at
`Stony Brook University <https://www.stonybrook.edu>`_.

This software suite is released under a dual-license model. For academic users,
individual users, or open-source software developers, you can use HIFIR under
the `AGPLv3+ <https://www.gnu.org/licenses/agpl-3.0.en.html>`_ license free of
charge for research and evaluation purpose. For commercial users, separate
commercial licenses are available through the Stony Brook University.
For inqueries regarding commercial licenses, please contact
Prof. Xiangmin Jiao at xiangmin.jiao@stonybrook.edu.

How to Cite ``hifir4py``
------------------------

If you use HIFIR (including ``hifir4py``) in your research for nonsingular
systems, please cite the following paper.

.. code-block:: bibtex

    @article{chen2021hilucsi,
        author  = {Chen, Qiao and Ghai, Aditi and Jiao, Xiangmin},
        title   = {{HILUCSI}: Simple, robust, and fast multilevel {ILU} for
                   large-scale saddle-point problems from {PDE}s},
        journal = {Numer. Linear Algebra Appl.},
        year    = {2021},
        doi     = {10.1002/nla.2400}
    }

If you use our work in solving ill-conditioned and singular systems, we
recommend you to cite the following papers.

.. code-block:: bibtex

    @article{jiao2020approximate,
        author  = {Xiangmin Jiao and Qiao Chen},
        journal = {arxiv},
        title   = {Approximate generalized inverses with iterative refinement
                  for $\epsilon$-accurate preconditioning of singular systems},
        year    = {2020},
        note    = {arXiv:2009.01673}
    }

    @article{chen2021hifir,
        author  = {Chen, Qiao and Jiao, Xiangmin},
        title   = {{HIFIR}: Hybrid incomplete factorization with iterative
                   refinement for preconditioning ill-conditioned and singular
                   Systems},
        journal = {arxiv},
        year    = {2021},
        note    = {arXiv:2106.09877}
    }

Contacts
--------

1. Qiao (Chiao) Chen, <qiao.chen@stonybrook.edu>, <benechiao@gmail.com>
2. Xiangmin Jiao, <xiangmin.jiao@stonybrook.edu>
