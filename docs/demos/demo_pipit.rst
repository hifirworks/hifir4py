Demo for using ``hifir4py`` with its PIPIT solver
=================================================

In this example, we show how to use ``hifir4py`` HIFIR-preconditioned
PIPIT solver for seeking the pseudoinverse solution of a linear
elasticity system with pure traction boundary conditions discretized by
FEM. Note that this system has six-dimensional nullspace.

.. code:: ipython3

    import numpy as np
    from scipy.io import loadmat
    from hifir4py import *

.. code:: ipython3

    # load the MATFILE from scipy.io
    f = loadmat("demo_inputs/LE_4.mat")
    A = f["A"]
    b = f["b"].reshape(-1)

Let's show some basic information of the system, including shape, nnz,
and leading block symmetry

.. code:: ipython3

    # A is scipy.sparse.csr_matrix
    print("The system shape is {}, where the nnz is {}".format(A.shape, A.nnz))


.. parsed-literal::

    The system shape is (2295, 2295), where the nnz is 83997


Now, we directly call ``pipit_hifir`` to seek for the pseudoinverse
solution with default settings, i.e., default parameters in HIF,
:math:`\text{rtol}=10^{-12}` for the null-space residual tolerance, and
:math:`\text{rtol}=10^{-6}` for the least-squares solution (by
HIF-preconditioned GMRES) relative residual tolerance; the ``restart``
and ``maxit`` are set to be 30 and 500, respectively.

.. code:: ipython3

    x, ns, flag, stats = ksp.pipit_hifir(A, b, 6)  # 6 is the null-space dimension


.. parsed-literal::

    HIF factorization finished in 0.1743s.
    
    Starting computing left nullspace...
    Finished left nullspace computation with total 46 GMRES iterations
    and total 75 inner refinements in 0.463s.
    
    Starting GMRES for least-squares solution...
    Preconditioned provided as input.
    Starting GMRES iterations...
    Computed solution in 7 iterations and 0.02111s.
    
    System is numerically symmetric; let vs=us.


We know analyze the accuracy of the null-space components. Note that
since this system is symmetric (thus range-symmetric).

.. code:: ipython3

    from scipy.sparse.linalg import norm as spnorm
    
    vs = ns["vs"]
    av = A.dot(vs)
    av_res = np.asarray([np.linalg.norm(av[i]) for i in range(6)]) / spnorm(A)
    print("relative residual of the six-dimensional nullspace are:", av_res)


.. parsed-literal::

    relative residual of the six-dimensional nullspace are: [6.44680278e-13 7.25437491e-13 2.64731149e-14 6.97660024e-13
     6.79220534e-13 1.09590900e-14]

