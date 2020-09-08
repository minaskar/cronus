============
Advanced Use
============

Log Likelihood Function
=======================


Parameter File
==============

The parameter file can generally include more information than the options presented in the :doc:`quickstart` page.

Likelihood
----------

Usually the argument of the Log Likelihood function is a *1D numpy array* but we can also use a dictionary instead.
To do so we need to add the ``dictionary: True`` option to the Likelihood block, for instance:

.. code:: yaml

    Likelihood:
      path: path/to/logprob.py
      function: log_likelihood
      dictionary: True


Parameters
----------

Every parameter needs to be either fixed or free:

- For fixed parameters we need to specify their value in Parameter block (i.e. parameter ``a`` in the following example).

- For free parameters we need to specify a prior instead. So far, only ``uniform`` and ``normal`` priors are supported.
  For a ``uniform`` prior we need to specify the uniform interval ``(min, max)`` (i.e. parameter ``b`` in the following
  example). For a ``normal`` prior we need to specify the mean ``loc`` and standard deviation ``scale`` (i.e. parameter
  ``c`` in the following example).

.. code:: yaml

    Parameters:
      a:
        fixed: 1.0
      b:
        prior:
          type: uniform
          min: -1.0
          max: 1.0
      c:
        prior:
          type: normal
          loc: 0.0
          scale: 1.0


Sampler
-------

``cronus`` supports three different samplers, ``zeus`` (Default), ``emcee``, and ``dynesty``. The prefered sampler can be specified
using the ``name`` option in the ``Sampler`` section of the parameter file, for instance:

.. code:: yaml

    Sampler:
      name: zeus
      ...

When either ``zeus`` or ``emcee`` is used as the prefered sampler then the following options are available:

- ``ndim`` is the total number of parameters/dimensions.
- ``nwalkers`` is the total number of walkers (i.e. internal parallel chains for zeus or emcee). This number needs to
  be at least twice the number of free parameters.
- ``nchains`` is the number of parallel chains, we recommend at least two and preferably 4 to get good estimate of the
  *Gelman-Rubin* diagnostic.
- ``ncheck`` specifies the number of steps after which the samples are saved and the *Convergence Criteria* are assessed.
  The default value is 100 which means that the samples are saved and convergence is diagnosed every 100 steps.
- ``maxiter`` specifies the maximum number of iterations (Default is inf).
- ``miniter`` specifies the minimum number of iterations (Default is 0).
- ``maxcall`` specifies the maximum number of Log Likelihood evalluations/calls (Default is inf).
- ``initial`` controls the initialization of the walker positions. The available options are: ``ellipse`` (this is a small
  ellipse around the *Maximum a posteriori* estimate, this is the default and recommended choice), ``laplace`` (sample the
  initial positions of the walkers from the *Laplace approximation* of the posterior distribution), and ``prior`` (sample
  the initial positions from the prior distribution, not the best choice).
- ``thin`` is the thinning rate for the chains (i.e. if ``thin=5`` then save every 5th element to the chain). This can
  significantly reduce the size of the output files if the autocorrelation time of the chain is large. The default value is 1.


When ``dynesty`` is used as the prefered sampler then the following options are available:

- ``ndim`` is the total number of parameters/dimensions.
- ``bound``
- ``dlogz``
- ``maxiter`` specifies the maximum number of iterations (Default is inf).
- ``maxcall`` specifies the maximum number of Log Likelihood evalluations/calls (Default is inf).
- ``pfrac``


Diagnostics
-----------

So far ``cronus`` includes two distinct convergence diagnostics, the Gelman-Rubin statistic and the Autocorrelation Time test.
Their combination seems to work well in Astrophysical and Cosmological likelihoods.

Lets see how one can customize the thresholds of those criteria:

- Either of them can be turned off or on (Default) using the ``use`` argument.
- ``|R_hat - 1| < epsilon`` is the threshold for the *Potential Scale Reduction Factor* (PSRF). We recommend to use a
  value of ``epsilon`` that it is smaller than 0.05 (Default).
- In terms of the *Integrated Autocorrelation Time* (IAT) we provide two criteria, if the chain is longer than ``nact = 20``
  (Default) times the estimated IAT and the IAT has changed less than ``dact = 3%`` (Default) the criteria are satisfied. If both
  *Gelman-Rubin* and IAT criteria are satisfied then sampling stops.

All of the diagnostic options can be seen here:

.. code:: yaml

    Diagnostics:
      Gelman-Rubin:
        use: True
        epsilon: 0.05
      Autocorrelation:
        use: True 
        nact: 20
        dact: 0.03


Output
------

The only option of the ``Output`` block is a directory path in which the samples/results will be saved. If
the provided directory doesn't exist one will be created by ``cronus``. The default directory is the current one.

.. code:: yaml

    Output: path/to/output/folder/chains


Running **cronus**
==================

To run ``cronus``, given a parameter file ``file.yaml``, we execute the following command:

.. code:: bash

    $ mpiexec -n [nprocesses] cronus-run file.yaml

where, ``nprocesses`` is the number of available CPUs. Depending on the cluster you are using you may need to use
``mpirun`` or ``srun`` instead of ``mpiexec``.

.. note::

    For better performance we recommend to use a number of processes that can be divided by the number of chains ``nchains``.
    Ideally, we recommend to use ``nchains * (nwalkers/2 + 1)`` if available, there's no real computational benefit in using
    more than this.


Results
=======

**zeus** or emcee
-----------------

When either ``zeus`` or ``emcee`` is used as the prefered sampler then the results are saved as ``h5`` files.
There are as many ``h5`` files saved as the number of chains ``nchains``. Each file contains two datasets, one
called ``samples`` which constists of the samples as the name suggests, and one named ``logprob`` which includes
the respective values of the Log Posterior Distribution.

After a few seconds of running the following files will be created in the provided ``Output`` directory:

    .. code-block:: bash

        chains
            ├── chain_0.h5
            ├── chain_1.h5
            ├── ...
            └── chain_[nchains].h5

The files will iteratively be updated every few iterations.

.. note::

    You can access those results by doing:

        .. code:: Python

            import numpy as np
            import h5py

            with h5py.File('chains/chain_0.h5', "r") as hf:
                samples = np.copy(hf['samples'])
                logprob = np.copy(hf['logprob'])
    
    The shape of the samples array would be ``(Iteration, nwalkers, ndim)`` and the shape of the Log Posterior array will
    be ``(Iteration, nwalkers)``. You can easily *flatten* this, combining all the walkers into one chain and discarding
    the first half of the chain, by running:

        .. code:: Python

            nsamples, nwalkers, ndim_prime = np.shape(samples)

            samples_flat = samples[nsamples//2:].reshape(-1, ndim_prime)

            logprob_flat = logprob[nsamples//2:].reshape(-1, 1)

dynesty
-------

When ``dynesty`` is used as the sampler then the results are saved as a numpy ``npy`` format file. 