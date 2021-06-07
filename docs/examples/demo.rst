
Demo for using ``hifir4py``
===========================

In this example, we show how to use ``hifir4py`` HIFIR preconditioner
coupling with the built-in GMRES solver.. The example system is a
saddle-point formulation of 3D Stokes equation with Taylor-Hood
elements.

.. code:: ipython3

    from hifir4py import *
    import numpy as np

The matrix is stored by the HIFIR native binary format that is leading
symmetric block aware. It's worht noting that. The following code shows
how to load the matrix.

.. code:: ipython3

    # load matrix and leading block size m
    rowptr, colind, vals, shape, _ = read_hifir("demo_inputs/A.hilucsi")

Let's show some basic information of the system, including shape, nnz,
and leading block symmetry

.. code:: ipython3

    # A is scipy.sparse.csr_matrix
    print("The system shape is {}, where the nnz is {}".format(rowptr[-1], shape))


.. parsed-literal::

    The system shape is 44632, where the nnz is (2990, 2990)


The rhs vector can be directly loaded from ``numpy`` ASCII routine

.. code:: ipython3

    b = np.loadtxt('demo_inputs/b.txt')

.. code:: ipython3

    assert shape[0] == len(b)

Now, let's build the preconditioenr :math:`\boldsymbol{M}` with default
configurations.

.. code:: ipython3

    M = HIF()
    M.factorize(rowptr, colind, vals, shape=shape)


.. parsed-literal::

    
    =======================================================================
    |           Hybrid (Hierarchical) Incomplete Factorizations           |
    |                                                                     |
    | HIF is a package for computing hybrid (hierarchical) incomplete fa- |
    | ctorizations with nearly linear time complexity.                    |
    -----------------------------------------------------------------------
    
     > Package information:
    
    		* Copyright (C) The HIF AUTHORS
    		* Version: 1.0.0
    		* Built on: 11:33:49, Jun  7 2021
    
    =======================================================================
    
    Params (control parameters) are:
    
    tau_L                         0.000100
    tau_U                         0.000100
    kappa_d                       3.000000
    kappa                         3.000000
    alpha_L                       10.000000
    alpha_U                       10.000000
    rho                           0.500000
    c_d                           10.000000
    c_h                           2.000000
    N                             -1
    verbose                       info
    rf_par                        1
    reorder                       Auto
    spd                           0
    check                         yes
    pre_scale                     0
    symm_pre_lvls                 1
    threads                       0
    mumps_blr                     1
    fat_schur_1st                 0
    rrqr_cond                     0.000000
    pivot                         off
    gamma                         1.000000
    beta                          1000.000000
    is_symm                       0
    no_pre                        0
    
    perform input matrix validity checking
    
    enter level 1 (asymmetric).
    
    performing symm preprocessing with leading block size  2990... 
    preprocessing done with leading block size 2826...
    time: 0.00519876s
    preparing data variables...
    start Crout update...
    finish Crout update...
    	total deferrals=80
    	leading block size in=2990
    	leading block size out=2746
    	diff=244
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
    time: 0.0234758s
    computing Schur complement and assembling Prec...
    	=================================
    	the Schur compl. has good size
    	=================================
    splitting LB and freeing L took 0.00106249s.
    splitting UB and freeing U took 0.00108128s.
    applying dropping on L_E and U_F with alpha_{L,U}=10,10...
    nnz(L_E)=100643/78770, nnz(U_F)=100643/78770, time: 0.000971111s...
    using 4 for Schur computation...
    pure Schur computation time: 0.0108929s...
    nnz(S_C)=49836, nnz(L/L_B)=128738/28095, nnz(U/U_B)=128738/28095
    dense_thres{1,2}=265540/1500...
    converted Schur complement (S) to dense for last level...
    factorizing dense level by RRQR with cond-thres 2.72713e+10...
    successfully factorized the dense component...
    time: 0.0179087s
    
    finish level 1.
    
    input nnz(A)=44632, nnz(precs)=141018, ratio=3.15957
    
    multilevel precs building time (overall) is 0.0493023s


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

    solver = GMRES(M)

.. code:: ipython3

    x, iters = solver.solve(rowptr, colind, vals, b, shape=shape)


.. parsed-literal::

    - GMRES -
    rtol=1e-06
    restart/cycle=30
    maxiter=500
    flex-kernel: tradition
    init-guess: no
    
    Calling traditional GMRES kernel...
    Enter outer iteration 1...
      At iteration 1, relative residual is 5.01853e-06.
      At iteration 2, relative residual is 1.17415e-08.


.. code:: ipython3

    print('solver done, with {} iterations and residule is {}'.format(iters, solver.resids[-1]))


.. parsed-literal::

    solver done, with 2 iterations and residule is 1.3711755436588413e-07

