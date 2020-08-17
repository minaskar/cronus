import numpy as np
from zeus import ChainManager
import sys

class initialize_walkers:

    def __init__(self, params, logpost_fn):
        self.parameters = params['Parameters']

        sampler_info = params['Sampler']
        self.ndim =  sampler_info['ndim']
        self.nwalkers =  sampler_info['nwalkers']
        self.nchains =  sampler_info['nchains']

        self.logpost_fn = logpost_fn

        self.start = np.empty((self.nwalkers, self.ndim))

        
        init_samples = self.sample_prior()
        logps = np.array(list(map(logpost_fn, init_samples)))
        max_idx = np.argmax(logps)
        self.centre = init_samples[max_idx]

    
    def sample_prior(self, ninit=100):

        init_samples  = np.empty((ninit, self.ndim))

        for j in range(ninit):
            for i, p in enumerate(self.parameters):
                if self.parameters[p]['prior']['type'] == 'uniform':
                    init_samples[j,i] = np.random.uniform(self.parameters[p]['prior']['min'],
                                                          self.parameters[p]['prior']['max'])
                        
                elif self.parameters[p]['prior']['type'] == 'normal':
                    init_samples[j,i] = np.random.normal(self.parameters[p]['prior']['loc'],
                                                         self.parameters[p]['prior']['scale'])

        return init_samples


    def _get_walkers(self):
        
        for chain in range(self.nchains):
            for walker in range(self.nwalkers):
                for i, p in enumerate(self.parameters):
                    if self.parameters[p]['prior']['type'] == 'uniform':
                        self.start[chain, walker, i] = np.random.uniform(self.parameters[p]['prior']['min'],
                                                                         self.parameters[p]['prior']['max'])
                        
                    elif self.parameters[p]['prior']['type'] == 'normal':
                        self.start[chain, walker, i] = np.random.normal(self.parameters[p]['prior']['loc'],
                                                                        self.parameters[p]['prior']['scale'])

        return self.start


    def get_walkers(self):
        
        
        for walker in range(self.nwalkers):
            for i, p in enumerate(self.parameters):
                if self.parameters[p]['prior']['type'] == 'uniform':
                    width = self.parameters[p]['prior']['max'] - self.parameters[p]['prior']['min']
                    while True:
                        self.start[walker, i] = self.centre[i] + np.random.uniform(-0.5, 0.5) * width * 0.01
                        if self.start[walker, i] > self.parameters[p]['prior']['min'] and self.start[walker, i] < self.parameters[p]['prior']['max']:
                            break
                    
                        
                elif self.parameters[p]['prior']['type'] == 'normal':
                    self.start[walker, i] = np.random.normal(self.centre[i],
                                                             0.01* self.parameters[p]['prior']['scale'])

        return self.start
