===========
Quick Start
===========

Overview
========

The main purpose of ``cronus`` is to facilitate large-scale Bayesian Inference (e.g. MCMC or NS) in modern
super-computing environments. ``cronus`` utilises ``MPI`` to efficiently distribute the tasks to multiple
nodes. Another important feature of ``cronus`` is its integrated and automated suite of *Convergence Diagnostics*.

Before we go into detail about how to use ``cronus`` let us first discuss the way it works in a higher level.
``cronus`` accepts as an input a parameter file that specifies the following:

- The Python file that contains the definition of the Log Likelihood function,
- A set of priors and/or fixed values for the different parameters of the model that enters the Log Likelihood function,
- A set of parameters that configure the MCMC/NS sampler (e.g. number of walkers), those are usually trivial to define.
- A few threshold values for the *Convergence Diagnostics*,
- The path/directory for the results to be saved in.

Once a parameter file is provided, ``cronus`` efficiently distributes the sampling tasks to all available CPUs and runs
until Convergence is reached. The results are saved iteratively so that the researcher can monitor the progress.

.. figure:: ./flow.png

Let us present here a simple example that will help illustrate the basic features and capabilities of ``cronus``.

Log Likelihood Function
=======================

The first thing we need to do is to create a Python file in which we define the Log Likelihood function. There is
no real restricton to this. The model itself can be computed in any programming language (e.g. C, C++, Fortran) and
the Log Likelihood can be a Python wrapper for this. In this example we will define a simple 3-dimensional Normal
distribution with a diagonal covariance matrix.

.. code:: python

    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    import numpy as np 

    ivar = 1.0 / np.random.rand(3)

    def log_likelihood(x):
        return - 0.5 * np.sum(ivar * x**2.0)

We then save the file as ``logprob.py``.

.. note::

    The important thing to note here is that the function accepts a single argument ``x``. If your Log Likelihood 
    requires more than one argument (e.g. data, covariance, etc.) we recommend to make those global like we did with
    the ``ivar`` array in the aforementioned example.

.. note::

    Some builds of NumPy (including the version included with Anaconda) will automatically parallelize some
    operations using something like the MKL linear algebra. This can cause problems when used with the
    parallelization methods described here so it can be good to turn that off (by setting the environment
    variable ``OMP_NUM_THREADS=1``, for example).

    .. code:: python

        import os
        os.environ["OMP_NUM_THREADS"] = "1"


Parameter File
==============

The next step is to create the  parameter file that we will call ``file.yaml``:

.. code:: yaml

    Likelihood:
      path: logprob.py
      function: log_likelihood

    Parameters:
      a:
        prior:
          type: uniform
          min: -10.0
          max: 10.0
      b:
        fixed: 1.0
      c:
        prior:
          type: normal
          loc: 1.0
          scale: 1.0

    Sampler:
      ndim: 3
      nwalkers: 10
      nchains: 4

    Diagnostics:
      Gelman-Rubin:
        epsilon: 0.05
      Autocorrelation:
        nact: 20
        dact: 0.03

    Output: chains

You can see the following *sections* in the parameter file:

- The ``Likelihood`` section which includes information about the path of the Log Likelihood function
  (i.e. both the directory/filename and the name of the function).
- The ``Parameters`` section which includes the priors of fixed values for each parameter of the model.
- The ``Sampler`` block which includes a few hyper-parameter values for the Sampler. Here ``ndim`` is the number of
  parameters/dimensions, ``nwalkers`` the number of parallel walkers of the ensemble (needs to be at least twice the
  number of free parameters), and ``nchains`` is the number of parallel chains. By default ``cronus`` relies on ``zeus``
  to do all the heavy-lifting, but you can also specify other samplers (see the :doc:`advanced` page for more information).
- The ``Diagnostics`` block is where we define the thresholds for the various *Convergence Diagnostics*. In this case 
  ``|R_hat - 1| < epsilon`` is the threshold for the *Potential Scale Reduction Factor* (PSRF). In terms of the
  *Integrated Autocorrelation Time* (IAT) we provide two criteria, if the chain is longer than ``nact = 20`` times the 
  estimated IAT and the IAT has changed less than ``dact = 3%`` the criteria are satisfied. If both *Gelman-Rubin* and
  IAT criteria are satisfied then sampling stops.
- The ``Output`` option specifies the output directory for the results to be saved in. If there's no such directory then
  ``cronus`` will build one.

For more information about the options in the parameter file please see the :doc:`advanced` page.

Run cronus
==========

To run this example go the directory where you saved ``file.yaml`` and do:

.. code:: bash

    $ mpiexec -n 8 cronus-run file.yaml

Here we used 8 CPUs.

Results
=======

After a few seconds the following files will be created in the provided ``Output`` directory:

    .. code-block:: bash

        chains
            ├── chain_0.h5
            ├── chain_1.h5
            ├── chain_2.h5
            └── chain_3.h5

The files will iteratively be updated every few iterations.

.. note::

    You can access those results by doing:

        .. code:: Python

            import numpy as np
            import h5py

            with h5py.File('chains/chain_0.h5', "r") as hf:
                data = np.copy(hf['samples'])
    
    The shape of the samples array would be ``(Iteration, nwalkers, ndim)``.
    You can easily *flatten* this, combining all the walkers into one chain, by running:

        .. code:: Python

            data_flat = data.reshape(-1, ndim)