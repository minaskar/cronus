import numpy as np

class initialize_walkers:

    def __init__(self, params):
        self.parameters = params['Parameters']

        sampler_info = params['Sampler']
        self.ndim =  sampler_info['ndim']
        self.nwalkers =  sampler_info['nwalkers']
        self.nchains =  sampler_info['nchains']

        self.start = np.empty((self.nchains, self.nwalkers, self.ndim))


    def get_walkers(self):
        
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
