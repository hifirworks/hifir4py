*hifir4py*
==========

:Release: |release|
:Date: |today|

Introduction
------------

``hifir4py`` is the Python3 interface of the HIFIR (Hybrid Incomplete
Factorization with Iterative Refinement) preconditioner, which is originally
written in C++. HIFIR is desiable for preconditioning ill-conditioned
and (potentially) singular systems to seek least-squares and/or the
pseudoinverse solutions.

Installation
------------

One can install ``hifir4py`` using ``pip``, i.e.,

.. code-block:: sh

    pip3 install hifir4py --user

In addition, refer to `<https://github.com/hifirworks/hifir4py/releases>`_ for
archives of different versions.

For Windows users, we recommend using `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/>`_
or `Cygwin <https://www.cygwin.com/>`_.

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

    @Article{chen2021hilucsi,
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

    @Article{jiao2020approximate,
        author  = {Xiangmin Jiao and Qiao Chen},
        journal = {arxiv},
        title   = {Approximate generalized inverses with iterative refinement
                  for $\epsilon$-accurate preconditioning of singular systems},
        year    = {2020},
        note    = {arXiv:2009.01673}
    }

    @Article{chen2021hifir,
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

1. Qiao Chen, qiao.chen@stonybrook.edu
2. Xiangmin Jiao, xiangmin.jiao@stonybrook.edu

API Reference
-------------

.. toctree::
    :maxdepth: 1

    hif_api
    ksp_api


Examples
--------

.. toctree::
    :maxdepth: 1

    demos/demo_apply
    demos/demo_scipy
    demos/demo_gmres
    demos/demo_pipit

.. include:: ../CHANGELOG.rst
