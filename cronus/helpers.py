import numpy as np 

import os
from ruamel.yaml import YAML

from .start import initialize_walkers
from .optimize import find_MAP

import time

from tqdm import tqdm

def damp_paramfile(output, params, rank=0):

    if rank==0:

        tag = params['Output']['tag']

        path = output + tag + "/"
        print('Output path : ', path, end='\r', flush=True)
        if not os.path.isdir(path):
            output = path
        else:
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


def save_results(output,results):

    with open(output+"results.dat", mode="w") as f:
        f.writelines(results.Summary)


def save_config(output, start, MAP, hessian, labels, cm, rank=0):
    start_all = cm.gather(start, root=0)

    if rank==0:
        conf = {'start':start_all, 'map' : MAP, 'hessian':hessian, 'labels':labels}
        np.save(output+'config.npy', conf)




class progress_bar:

    def __init__(self, rank=0, show=True):
        self.t0 = time.time()
        self.show = show
        if self.show and rank==0:
            self.progress_bar = tqdm(desc='Iter')

    def update(self, cnt, ncall, rhat, tau, dact, nact, rank=0):
        
        if self.show and rank==0:
            dt = time.time() - self.t0
            clock = time.strftime("%H:%M:%S", time.gmtime(dt))
    
            _eta = dt / (cnt/tau) * (nact - cnt/tau)
            eta = time.strftime("%H:%M:%S", time.gmtime(_eta))

            self.progress_bar.update(1)
            self.progress_bar.set_postfix(ordered_dict={'ncall':ncall,
                                                        'R-hat':round(rhat,4),
                                                        'act':tau,
                                                        'nact':int(cnt/tau),
                                                        'dact':dact,
                                                        'ETA':eta
                                                        })

    def close(self, rank=0):
        if self.show and rank==0:
            self.progress_bar.close()