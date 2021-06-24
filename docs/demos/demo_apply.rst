Demo for using ``hifir4py``
===========================

In this example, we show how to use ``hifir4py`` HIFIR preconditioner in
its multilevel triangular solve and matrix-vector multiplication, which
are the core operations in applying a preconditioner.

.. code:: ipython3

    import numpy as np
    from scipy.io import loadmat
    from hifir4py import *

.. code:: ipython3

    # load the MATFILE from scipy.io
    f = loadmat("demo_inputs/data.mat")
    A = f["A"]
    b = f["b"].reshape(-1)

Let's show some basic information of the system, including shape, nnz,
and leading block symmetry

.. code:: ipython3

    # A is scipy.sparse.csr_matrix
    print("The system shape is {}, where the nnz is {}".format(A.shape, A.nnz))


.. parsed-literal::

    The system shape is (2990, 2990), where the nnz is 44632


Factorize HIF
-------------

Now, let's build the preconditioenr :math:`\boldsymbol{M}` with more
aggressive options, i.e. ``droptol`` for L and U factors is 1e-2,
``condest`` for L, U, and D is 5, and :math:`\alpha` for L and U is 3.

.. code:: ipython3

    M = HIF(
        A,
        tau_L=0.01,
        tau_U=0.01,
        kappa=5.0,
        kappa_d=5.0,
        alpha_L=3.0,
        alpha_U=3.0,
    )


.. parsed-literal::

    =======================================================================
    |           HIF: Hybrid Incomplete Factorization                      |
    |                                                                     |
    | HIF is a package for computing hybrid incomplete factorization      |
    | with near linear time complexity.                                   |
    -----------------------------------------------------------------------
    
     > Package information:
    
    	* Copyright (C) 2019--2021 NumGeom Group at Stony Brook University
    	* Version: 0.1.0
    
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
    reorder                       Off
    spd                           0
    check                         yes
    pre_scale                     0
    symm_pre_lvls                 1
    threads                       0
    fat_schur_1st                 0
    rrqr_cond                     0.000000
    pivot                         auto
    gamma                         1.000000
    beta                          1000.000000
    is_symm                       0
    no_pre                        0
    
    perform input matrix validity checking
    
    enter level 1 (asymmetric).
    
    performing symm preprocessing with leading block size  2990... 
    preprocessing done with leading block size 2826...
    time: 0.00360131s
    preparing data variables...
    start Crout update...
    finish Crout update...
    	total deferrals=1
    	leading block size in=2990
    	leading block size out=2825
    	diff=165
    	diag deferrals=0
    	inv-norm deferrals=1
    	drop ut=113670
    	space drop ut=22023
    	drop l=113670
    	space drop l=22023
    	min |kappa_u|=1
    	max |kappa_u|=4.77626
    	min |kappa_l|=1
    	max |kappa_l|=4.77626
    	max |d|=1
    time: 0.0237231s
    computing Schur complement and assembling Prec...
    	=================================
    	the Schur complement has good size
    	=================================
    splitting LB and freeing L took 0.000720598s.
    splitting UB and freeing U took 0.00078849s.
    applying dropping on L_E and U_F with alpha_{L,U}=3,3...
    nnz(L_E)=53031/30145, nnz(U_F)=53031/30145, time: 0.00131117s...
    using 4 for Schur computation...
    pure Schur computation time: 0.00542072s...
    nnz(S_C)=16953, nnz(L/L_B)=85269/32238, nnz(U/U_B)=85269/32238
    dense_thres{1,2}=61032/2000...
    converted Schur complement (S) to dense for last level...
    factorizing dense level by RRQR with cond-thres 2.72713e+10...
    successfully factorized the dense component...
    time: 0.0106012s
    
    finish level 1.
    
    input nnz(A)=44632, nnz(precs)=114848, ratio=2.57322
    
    multilevel precs building time (overall) is 0.0398237s


With the preconditioenr successfully been built, let's print out some
basic information

.. code:: ipython3

    print("M levels are {}, with nnz {}".format(M.levels, M.nnz))


.. parsed-literal::

    M levels are 2, with nnz 114848


Alternatively, one can use the following codes:

.. code:: python

    M = HIF()
    params=Params()
    params.tau = 1e-2  # equiv. to params["tau_L"]=params["tau_U"]=1e-2
    params.kappa = 5.0
    params.alpha = 3.0
    M.factorize(A, params=params)

or

.. code:: python

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

.. code:: ipython3

    x = M.apply(b)
    err = M.apply(x, op="M") - b
    print("norm2(err)/norm2(b) =", np.linalg.norm(err)/np.linalg.norm(b))


.. parsed-literal::

    norm2(err)/norm2(b) = 1.439076582138997e-17


.. code:: ipython3

    # Tranpose
    x = M.apply(b, op="SH")
    err = M.apply(x, op="MH") - b
    print("norm2(err)/norm2(b) =", np.linalg.norm(err)/np.linalg.norm(b))


.. parsed-literal::

    norm2(err)/norm2(b) = 1.4514835137900503e-17

