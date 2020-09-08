import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from ruamel.yaml import YAML
from pathlib import Path

from .mcmc import sampler
from .default import get_default
from .likelihood import import_loglikelihood

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

    # Fix parameters
    params = get_default(params)

    # Create Output folder
    try:
        os.makedirs(params['Output'])
    except:
        pass

    # Import Log Likelihood function from file
    loglike_fn = import_loglikelihood(params)

    # Run Inference
    sampler(params).run(loglike_fn)