.. _demo_gmres:

Demo for using ``hifir4py`` with built-in ``gmres_hif``
=======================================================

In this example, we show how to use ``hifir4py`` HIFIR preconditioner
coupling with the built-in ``gmres_hif`` solver. The example system is a
saddle-point formulation of 3D Stokes equation with Taylor-Hood
elements.

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


Now, let's build the preconditioenr :math:`\boldsymbol{M}` with more
aggressive options, i.e. ``droptol`` for L and U factors is 1e-2,
``condest`` for L, U, and D is 5, and :math:`\alpha` for L and U is 3.

.. code-block:: ipython3

    M = HIF()
    params = Params()
    params.tau = 0.01
    params.kappa = 5.0
    params.alpha = 3.0
    M.factorize(A, params=params)

With the preconditioenr successfully been built, let's print out some
basic information

.. code-block:: ipython3

    print("M levels are {}, with nnz {}".format(M.levels, M.nnz))


.. code-block:: text

    M levels are 2, with nnz 114848


Now, we solve with the built-in flexible GMRES solver in SciPy. Notice
that the GMRES in SciPy is left-preconditioned, which is not
recommended.

.. code-block:: ipython3

    
    x, flag, stats = ksp.gmres_hif(A, b, M=M)


.. code-block:: text

    Preconditioned provided as input.
    Starting GMRES iterations...
    Computed solution in 3 iterations and 0.07072s.


.. code-block:: ipython3

    print("solver done, flag={}, res={}".format(flag, np.linalg.norm(b-A.dot(x))/np.linalg.norm(b)))


.. code-block:: text

    solver done, flag=0, res=8.254808859114319e-07

