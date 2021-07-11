Demo for using ``hifir4py``
===========================

In this example, we show how to use ``hifir4py`` HIFIR preconditioner in
its multilevel triangular solve and matrix-vector multiplication, which
are the core operations in applying a preconditioner.

.. code-block:: ipython3

    import numpy as np
    from scipy.io import loadmat
    from hifir4py import *

.. code-block:: ipython3

    # load the MATFILE from scipy.io
    f = loadmat("demo_inputs/data.mat")
    A = f["A"]
    b = f["b"].reshape(-1)

Let's show some basic information of the system, including shape, nnz,
and leading block symmetry

.. code-block:: ipython3

    # A is scipy.sparse.csr_matrix
    print("The system shape is {}, where the nnz is {}".format(A.shape, A.nnz))


.. code-block:: text

    The system shape is (2990, 2990), where the nnz is 44632


Factorize HIF
-------------

Now, let's build the preconditioenr :math:`\boldsymbol{M}` with more
aggressive options, i.e. ``droptol`` for L and U factors is 1e-2,
``condest`` for L, U, and D is 5, and :math:`\alpha` for L and U is 3.

.. code-block:: ipython3

    M = HIF(
        A,
        tau_L=0.01,
        tau_U=0.01,
        kappa=5.0,
        kappa_d=5.0,
        alpha_L=3.0,
        alpha_U=3.0,
    )

With the preconditioenr successfully been built, let's print out some
basic information

.. code-block:: ipython3

    print("M levels are {}, with nnz {}".format(M.levels, M.nnz))


.. code-block:: text

    M levels are 2, with nnz 114848


Alternatively, one can use the following codes:

.. code-block:: python

    M = HIF()
    params=Params()
    params.tau = 1e-2  # equiv. to params["tau_L"]=params["tau_U"]=1e-2
    params.kappa = 5.0
    params.alpha = 3.0
    M.factorize(A, params=params)

or

.. code-block:: python

    M = HIF()
    M.factorize(A,
        tau_L=0.01,
        tau_U=0.01,
        kappa=5.0,
        kappa_d=5.0,
        alpha_L=3.0,
        alpha_U=3.0,
    )

Apply HIF
---------

We now consider applying ``M`` in triangular solve and matrix-vector
multiplication two modes.

.. code-block:: ipython3

    x = M.apply(b)
    err = M.apply(x, op="M") - b
    print("norm2(err)/norm2(b) =", np.linalg.norm(err)/np.linalg.norm(b))


.. code-block:: text

    norm2(err)/norm2(b) = 1.439076582138997e-17


.. code-block:: ipython3

    # Tranpose
    x = M.apply(b, op="SH")
    err = M.apply(x, op="MH") - b
    print("norm2(err)/norm2(b) =", np.linalg.norm(err)/np.linalg.norm(b))


.. code-block:: text

    norm2(err)/norm2(b) = 1.4514835137900503e-17

