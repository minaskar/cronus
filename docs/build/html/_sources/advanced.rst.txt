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


Output
------


Running **cronus**
==================


Results
=======

When either ``zeus`` or ``emcee`` is used as the prefered sampler then the results are saved as ``h5`` files.
There are as many ``h5`` files saved as the number of chains ``nchains``. Each file contains two datasets, one
called ``samples`` which constists of the samples as the name suggests, and one named ``logprob`` which includes
the respective values of the Log Posterior Distribution.