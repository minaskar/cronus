===============
Troubleshooting
===============

infiniband
==========

There seem to be some issues with some ``mpi4py`` features when used in a computing cluster with *infiniband*.
This leads to ``cronus`` to hang in an ``ìnfiniband`` multi-node setting.

OpenMPI
-------

If you are using ``OpenMPI`` you can try including the following command which in your jobscript.

.. code:: bash

    export OMPI_MCA_pml=ob1

This should disable the *infiniband* interface.

Intel MPI
---------

The mpi4py package is using matching probes ``(MPI_Mpobe)`` for the receiving function ``recv()`` instead of regular
``MPI_Recv`` operations per default. These matching probes from the ``MPI 3.0`` standard however are not supported
for all fabrics, which may lead to a hang in the receiving function.

Therefore, users are recommended to leverage the ``OFI`` fabric instead of ``TMI`` for ``Omni-Path`` systems. For the
``Intel MPI Library``, the configuration could look like the following environment variable setting:

.. code:: bash

    export I_MPI_FABRICS=ofi