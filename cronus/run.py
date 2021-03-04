import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from ruamel.yaml import YAML
from pathlib import Path

from .mcmc import sampler
from .default import get_default
from .likelihood import import_loglikelihood

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
    if rank==0:
        print('Importing likelihood....', flush=True)
    loglike_fn = import_loglikelihood(params)
    if rank==0:
        print('Likelihood imported successfully....', flush=True)

    # Run Inference
    sampler(params).run(loglike_fn, continue_mcmc=False)


def continue_script():

    # Read parameter file as an argument
    parser = argparse.ArgumentParser(description='Continue running chains')
    parser.add_argument("directory", help="directory of project folder e.g chains/run")
    args = parser.parse_args()
    name = args.directory
    if name[-1] != '/':
        name += '/'
    #print(name, flush=True)

    # Read yaml parameter file and extract information as a dictionary
    path = Path(name+'para.yaml')
    yaml = YAML(typ='safe')
    params = yaml.load(path)

    # Fix parameters
    params = get_default(params)
    params['Output']['directory'] = name
    
    # Import Log Likelihood function from file
    #print('Import likelihood....', end='\r', flush=True)
    if rank==0:
        print('Importing likelihood....', flush=True)
    loglike_fn = import_loglikelihood(params)
    if rank==0:
        print('Likelihood imported successfully....', flush=True)

    # Run Inference
    sampler(params).run(loglike_fn, continue_mcmc=True)