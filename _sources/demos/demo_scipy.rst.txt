Demo for using ``hifir4py`` with SciPy's GMRES
==============================================

In this example, we show how to use ``hifir4py`` HIFIR preconditioner
coupling with the built-in GMRES solver in SciPy. The example system is
a saddle-point formulation of 3D Stokes equation with Taylor-Hood
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
    params['tau_L'] = params['tau_U'] = 0.01
    params['kappa'] = params['kappa_d'] = 5.0
    params['alpha_L'] = params['alpha_U'] = 3
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

    from scipy.sparse.linalg import gmres

.. code-block:: ipython3

    iters = 0
    def counter(res=None):
        global iters
        iters += 1
        print("At iteration {}, residual is {}".format(iters, res))
    x, flag = gmres(A, b, M=M.to_scipy(), callback=counter)


.. code-block:: text

    At iteration 1, residual is 0.03726697689855636
    At iteration 2, residual is 0.004555930822500451
    At iteration 3, residual is 0.0005254851747732505
    At iteration 4, residual is 4.8775926768822654e-05
    At iteration 5, residual is 4.3329265915834745e-06
    At iteration 6, residual is 3.830896351600058e-07


.. code-block:: ipython3

    print("solver done, flag={}, res={}".format(flag, np.linalg.norm(b-A.dot(x))/np.linalg.norm(b)))


.. code-block:: text

    solver done, flag=0, res=3.7186961167705056e-07

