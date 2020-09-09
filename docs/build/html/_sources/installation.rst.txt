============
Installation
============

Requirements
============

``cronus`` is compatible with Python 3.6+. It requires ``numpy``, ``scipy``, ``mpi4py``, ``ìminuit``, ``h5py`` and ``zeus`` to run.
If you want to use ``cronus`` with either ``emcee`` or ``dynesty`` please make sure that you have
those installed too.

You can find information about how to install ``mpi4py`` and its prerequisites at  https://mpi4py.readthedocs.io/en/stable/install.html

Install using pip
=================

We recommend to use pip to install the latest stable version of  ``cronus``::

    pip install cronus-mcmc


Install from source
===================

Alternatively, install the latest version of ``cronus`` from source::

    git clone https://github.com/minaskar/cronus.git
    cd cronus
    pip install -r requirements.txt
    pip install .


Making sure that cronus is installed properly
=============================================

If everything went well, you should be able to import ``cronus`` in Python from anywhere in your directory structure::

    $ python -c "import cronus"

If you get an error message, something went wrong. Check twice the instructions above, try again, or contact us.

``cronus`` also installs some shell scripts. If everything went well, if you try to run in the shell ``cronus-run``, you
should get a message asking you for an input file, instead of a command not found error.

.. note::

    If you do get a command not found error, this means that the folder where your local scripts are installed has not
    been added to your path.

    To solve this on unix-based machines, look for the ``cronus-run`` script from your home and scratch folders with::

        $ find `pwd` -iname cronus-run -printf %h\\n

    in Linux, or::

        $ which -a cronus-run
    
    in Mac OS X.

    This should print the location of the script, e.g. ``/home/you/.local/bin``. Add::

        $ export PATH="/home/you/.local/bin":$PATH

    at the end of your ``~/.bashrc`` file, and restart the terminal or do ``source ~/.bashrc``. Alternatively, simply
    add that line to your cluster jobscripts just before calling ``cronus-run``.