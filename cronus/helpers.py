import numpy as np 

import os
from ruamel.yaml import YAML

from .start import initialize_walkers
from .optimize import find_MAP

import time

def damp_paramfile(output, params):
    
    nrun = 1
    while True:
        path = output + "run" + str(nrun) + "/"
        if os.path.isdir(path):
            nrun += 1
        else:
            output = path
            break
    
    os.makedirs(path)
    # Dump full parameter file
    yaml = YAML()
    with open(output + "para.yaml", mode='w') as f:
        yaml.dump(params, f)

    return output


def initialise_sampler(name, nwalkers, ndim, logprob, pool):

    if name == 'zeus':
        import zeus
        sampler = zeus.EnsembleSampler(nwalkers, ndim, logprob, pool=pool, verbose=False)
    elif name == 'emcee':
        import emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, pool=pool)

    return sampler


def create_gelmanrubin(output, free_labels):
    with open(output + 'GelmanRubin.dat', 'w') as f:
        header = 'Iter'
        for p in free_labels:
            header += '  R_' + p
        f.write(header+'\n')


def create_varnames(output, free_labels):
    with open(output + 'varnames.dat', 'w') as f:
        header = ''
        for p in free_labels:
            header += p + " "
        f.write(header[:-1])


def create_IAT(output, free_labels, rank):
    with open(output + 'IAT_'+ str(rank) +'.dat', 'w') as f:
        header = 'Iter'
        for p in free_labels:
            header += '  tau_' + p
        f.write(header+'\n')


def get_starting_positions(params, distribution, cm):

    ensemble = initialize_walkers(params, distribution)

    x0, f0, h0 = find_MAP(params, distribution, ensemble.bounds)

    x0s = cm.allgather(x0)
    f0s = cm.allgather(f0)
    h0s = cm.allgather(h0)

    x0_best = x0s[np.argmin(f0s)]
    h0_best = h0s[np.argmin(f0s)]

    start = ensemble.get_walkers(x0_best, h0_best)

    return start, x0_best, h0_best


def append_IAT(output, cnt, act, rank):

    with open(output + 'IAT_' + str(rank) + '.dat', 'a') as f:
        f.write(str(cnt)  + " " + " ".join(str(round(t,4)) for t in act)+"\n")


def append_GR(output, cnt, rhat):

    with open(output + 'GelmanRubin.dat', 'a') as f:
        f.write(str(cnt) + " " + " ".join(str(round(r,4)) for r in rhat)+"\n")


def print_status(t0, cnt, ncall, rhat, taus, nact, deltas, dact):

    clock = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
    print(clock, '| Iter:', cnt, '| ncall:', ncall, '| R-hat:', round(np.max(rhat),4),
        '| act:', np.max(taus), '| nact:', int(cnt/np.max(taus)), '<', nact,
        '| dact:', np.max(deltas), '<', dact, end='\r', flush=True)


def save_results(output,results):

    with open(output+"results.dat", mode="w") as f:
        f.writelines(results.Summary)