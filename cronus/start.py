import numpy as np
from zeus import ChainManager

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

        self.centre = np.array([ 1.43563590e+00, -2.79228917e-02,  1.01513786e+01, -2.94868355e+02,
                                 2.45089136e+03, -3.57808097e+03,  1.85639553e-01, -2.42260305e-01,
                                -1.16837933e+03,  1.30581940e+03,  6.21201014e+01,  9.17828705e-01,
                                -7.10687852e+01,  7.39951459e+02, -4.96254867e+03,  1.11538809e+04,
                                 8.99609731e-01, 4.79761397e-01, 7.86391667e+00, 1.50709982e+00,
                                 1.06320700e+00,  1.00022316e+00])

    
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
