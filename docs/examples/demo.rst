
Demo for using ``hilucsi4py``
=============================

In this example, we show how to use ``hilucs4py`` MILU preconditioner
coupling with the built-in FGMRES solver.. The example system is a
saddle-point formulation of 3D Stokes equation with Taylor-Hood
elements.

.. code:: ipython3

    from hilucsi4py import *
    import numpy as np

The matrix is stored by the HILUCSI native binary format that is leading
symmetric block aware. It's worht noting that. The following code shows
how to load the matrix.

.. code:: ipython3

    # load matrix and leading block size m
    rowptr, colind, vals, shape, m = read_hilucsi('demo_inputs/A.hilucsi')

Let's show some basic information of the system, including shape, nnz,
and leading block symmetry

.. code:: ipython3

    # A is scipy.sparse.csr_matrix
    print('The system shape is {}, where the nnz is {}, leading block is {}'.format(rowptr[-1], shape, m))


.. parsed-literal::

    The system shape is 44632, where the nnz is (2990, 2990), leading block is 0


The rhs vector can be directly loaded from ``numpy`` ASCII routine

.. code:: ipython3

    b = np.loadtxt('demo_inputs/b.txt')

.. code:: ipython3

    assert shape[0] == len(b)

Now, let's build the preconditioner :math:`\boldsymbol{M}` with default
configurations.

.. code:: ipython3

    M = HILUCSI()
    M.factorize(rowptr, colind, vals, shape=shape)


.. parsed-literal::

    
    =======================================================================
    |    Hierarchical ILU Crout with Scalability and Inverse Thresholds   |
    |                                                                     |
    | HILUCSI is a package for computing multilevel incomplete LU factor- |
    | ization with nearly linear time complexity. In addition, HILUCSI    |
    | can also be very robust.                                            |
    -----------------------------------------------------------------------
    
     Package information:
    
    		Copyright (C) The HILUCSI AUTHORS
    		Version: 1.0.0
    		Built on: 23:01:07, Jul 12 2019
    
    =======================================================================
    
    Options (control parameters) are:
    
    tau_L                         0.000100
    tau_U                         0.000100
    tau_d                         3.000000
    tau_kappa                     3.000000
    alpha_L                       10
    alpha_U                       10
    rho                           0.500000
    c_d                           10.000000
    c_h                           2.000000
    N                             -1
    verbose                       info
    rf_par                        1
    reorder                       Auto
    saddle                        1
    pre_reorder                   Off
    pre_reorder_lvl1              1
    matching                      Auto
    pre_scale                     0
    symm_pre_lvls                 1
    
    perform input matrix validity checking
    
    enter level 1 (asymmetric).
    
    performing symm preprocessing with leading block size 2990...
    preprocessing done with leading block size 2826...
    time: 0.00306198s
    preparing data variables...
    start Crout update...
    finish Crout update...
    	total deferrals=80
    	leading block size in=2826
    	leading block size out=2746
    	diff=80
    	diag deferrals=14
    	inv-norm deferrals=66
    	drop ut=35562
    	space drop ut=60
    	drop l=35562
    	space drop l=60
    	min |kappa_u|=1
    	max |kappa_u|=2.9844
    	min |kappa_l|=1
    	max |kappa_l|=2.9844
    	max |d|=1
    time: 0.023357s
    computing Schur complement and assembling Prec...
    applying dropping on L_E and U_F with alpha_{L,U}=10,10...
    nnz(L_E)=100643/78770, nnz(U_F)=100643/78770...
    nnz(S_C)=49836, nnz(L/L_B)=128738/28095, nnz(U/U_B)=128738/28095
    dense_thres{1,2}=265540/1500...
    converted Schur complement (S) to dense for last level...
    successfully factorized the dense component...
    time: 0.0252393s
    
    finish level 1.
    
    input nnz(A)=44632, nnz(precs)=141018, ratio=3.15957
    
    multilevel precs building time (overall) is 0.0524915s


With the preconditioenr successfully been built, let's print out some
basic information

.. code:: ipython3

    print('M levels are {}, with nnz {}'.format(M.levels, M.nnz))


.. parsed-literal::

    M levels are 2, with nnz 141018


Now, we solve with the built-in flexible GMRES solver, with default
configurations, i.e. restart is 30, relative convergence tolerance is
1e-6, and maximum allowed iterations are 500.

.. code:: ipython3

    solver = FGMRES(M)

.. code:: ipython3

    x, iters = solver.solve(rowptr, colind, vals, b, shape=shape)


.. parsed-literal::

    - FGMRES -
    rtol=1e-06
    restart=30
    maxiter=500
    kernel: tradition
    init-guess: no
    trunc: no
    
    Calling traditional GMRES kernel...
    Enter outer iteration 1...
      At iteration 1 (inner:1), relative residual is 5.01853e-06.
      At iteration 2 (inner:1), relative residual is 1.17415e-08.


.. code:: ipython3

    print('solver done, with {} iterations and residual is {}'.format(iters, solver.resids[-1]))


.. parsed-literal::

    solver done, with 2 iterations and residual is 1.174147139978338e-08

