
Demo for using ``hifir4py`` with user ``Params`` and SciPy sparse matrix
========================================================================

In this example, we show how to use ``hifir4py`` HIFIR preconditioner
coupling with the built-in GMRES solver.. The example system is a
saddle-point formulation of 3D Stokes equation with Taylor-Hood
elements.

.. code:: ipython3

    from scipy.io import loadmat
    from hifir4py import *
    import numpy as np

.. code:: ipython3

    # load the MATFILE from scipy.io
    f = loadmat("demo_inputs/matlab.mat")
    A = f["A"]
    b = f["b"].reshape(-1)

Let's show some basic information of the system, including shape, nnz,
and leading block symmetry

.. code:: ipython3

    # A is scipy.sparse.csr_matrix
    print('The system shape is {}, where the nnz is {}'.format(A.shape, A.nnz))


.. parsed-literal::

    The system shape is (2990, 2990), where the nnz is 44632


Now, let's build the preconditioenr :math:`\boldsymbol{M}` with more
aggressive options, i.e. ``droptol`` for L and U factors is 1e-2,
``condest`` for L, U, and D is 5, and :math:`\alpha` for L and U is 3.

.. code:: ipython3

    M = HIF()
    params = Params()
    params['tau_L'] = params['tau_U'] = 0.01
    params['kappa'] = params['kappa_d'] = 5.0
    params['alpha_L'] = params['alpha_U'] = 3
    M.factorize(A, params=params)


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
    
    tau_L                         0.010000
    tau_U                         0.010000
    kappa_d                       5.000000
    kappa                         5.000000
    alpha_L                       3.000000
    alpha_U                       3.000000
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
    time: 0.00565477s
    preparing data variables...
    start Crout update...
    finish Crout update...
    	total deferrals=9
    	leading block size in=2990
    	leading block size out=2817
    	diff=173
    	diag deferrals=0
    	inv-norm deferrals=9
    	drop ut=47645
    	space drop ut=7337
    	drop l=47645
    	space drop l=7337
    	min |kappa_u|=1
    	max |kappa_u|=4.98881
    	min |kappa_l|=1
    	max |kappa_l|=4.98881
    	max |d|=1
    time: 0.0155806s
    computing Schur complement and assembling Prec...
    	=================================
    	the Schur compl. has good size
    	=================================
    splitting LB and freeing L took 0.000570666s.
    splitting UB and freeing U took 0.00051511s.
    applying dropping on L_E and U_F with alpha_{L,U}=3,3...
    nnz(L_E)=51977/29167, nnz(U_F)=51977/29167, time: 0.00122208s...
    using 4 for Schur computation...
    pure Schur computation time: 0.00539931s...
    nnz(S_C)=18595, nnz(L/L_B)=70610/18633, nnz(U/U_B)=70610/18633
    dense_thres{1,2}=63168/1500...
    converted Schur complement (S) to dense for last level...
    factorizing dense level by RRQR with cond-thres 2.72713e+10...
    successfully factorized the dense component...
    time: 0.010071s
    
    finish level 1.
    
    input nnz(A)=44632, nnz(precs)=90664, ratio=2.03137
    
    multilevel precs building time (overall) is 0.0328081s


With the preconditioenr successfully been built, let's print out some
basic information

.. code:: ipython3

    print('M levels are {}, with nnz {}'.format(M.levels, M.nnz))


.. parsed-literal::

    M levels are 2, with nnz 90664


Now, we solve with the built-in flexible GMRES solver, with default
configurations, i.e. restart is 30, relative convergence tolerance is
1e-6, and maximum allowed iterations are 500.

.. code:: ipython3

    solver = GMRES(M)

.. code:: ipython3

    x, iters = solver.solve(A, b)


.. parsed-literal::

    - GMRES -
    rtol=1e-06
    restart/cycle=30
    maxiter=500
    flex-kernel: tradition
    init-guess: no
    
    Calling traditional GMRES kernel...
    Enter outer iteration 1...
      At iteration 1, relative residual is 0.000216346.
      At iteration 2, relative residual is 5.54575e-06.
      At iteration 3, relative residual is 4.62893e-07.


.. code:: ipython3

    print('solver done, with {} iterations and residule is {}'.format(iters, solver.resids[-1]))


.. parsed-literal::

    solver done, with 3 iterations and residule is 5.405694712085073e-06

