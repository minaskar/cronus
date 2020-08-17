import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from ruamel.yaml import YAML
from pathlib import Path

from .likelihood import import_loglikelihood
from .prior import import_logprior
from .posterior import define_logposterior
from .mcmc import sampler
from .start import initialize_walkers

import numpy as np

import sys

def run_script():

    # Read parameter file as an argument
    parser = argparse.ArgumentParser(description='Run some chains')
    parser.add_argument("paramfile", help="filename of parameter file")
    args = parser.parse_args()
    name = args.paramfile
    if name[-5:] != '.yaml':
        name += '.yaml'

    # Read yaml parameter file and extract information as a dictionary
    path = Path(name)
    yaml = YAML(typ='safe')
    params = yaml.load(path)

    # Import loglikelihood from parameter file information
    loglike_fn = import_loglikelihood(params)

    # Import logprior from parameter file information
    logprior_fn = import_logprior(params).get_logprior

    # Define logposterior
    logpost_fn = define_logposterior(params, loglike_fn, logprior_fn).get_logposterior

    # Initialize walkers
    ensemble = initialize_walkers(params, logpost_fn)
    p0 = ensemble.get_walkers()

    # Run MCMC
    sampler(params).run_mcmc(logpost_fn, p0)