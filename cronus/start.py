import numpy as np

from .optimize import find_MAP


class initialize_walkers:

    def __init__(self, params, distribution):
        self.params = params
        self.parameters = params['Parameters']

        self.distribution = distribution

        self.sampler_info = params['Sampler']
        self.ndim = self.sampler_info['ndim']
        self.nwalkers = self.sampler_info['nwalkers']

        self.get_logprior = distribution.get_logprior

        self.low = np.empty(distribution.nfree)
        self.high = np.empty(distribution.nfree)
        self.bounds = self.find_bounds()

    
    def find_bounds(self):
        bounds = []
        for i, p in enumerate(self.distribution.free_labels):
            
            if self.parameters[p]['prior']['type'] == 'uniform':
                bounds.append([self.parameters[p]['prior']['min'], self.parameters[p]['prior']['max']])
                self.low[i] = self.parameters[p]['prior']['min']
                self.high[i] = self.parameters[p]['prior']['max']
                    
            elif self.parameters[p]['prior']['type'] == 'normal':
                loc = self.parameters[p]['prior']['loc']
                scale = self.parameters[p]['prior']['scale']
                bounds.append([loc-5.0*scale, loc+5.0*scale])
                self.low[i] = loc-5.0*scale
                self.high[i] = loc+5.0*scale
        return bounds


    def get_ellipse(self, x0):

        sigma = self.high - self.low

        start = np.empty((self.nwalkers, self.distribution.nfree))
        
        for w in range(self.nwalkers):
            while True:
                pos = np.random.randn(self.distribution.nfree)*sigma*0.01 + x0
                if np.isfinite(self.get_logprior(pos)):
                    start[w] = pos
                    break

        return start


    def get_laplace(self, x0, hess_inv):

        start = np.empty((self.nwalkers, self.distribution.nfree))

        for w in range(self.nwalkers):
            while True:
                pos = np.random.multivariate_normal(x0, hess_inv, check_valid='ignore')
                if np.isfinite(self.get_logprior(pos)):
                    start[w] = pos
                    break

        return start


    def get_prior(self):

        start = np.empty((self.nwalkers, self.distribution.nfree))
        
        for walker in range(self.nwalkers):
            for i, p in enumerate(self.parameters):
                if self.parameters[p]['prior']['type'] == 'uniform':
                    start[walker, i] = np.random.uniform(self.parameters[p]['prior']['min'],
                                                                     self.parameters[p]['prior']['max'])
                        
                elif self.parameters[p]['prior']['type'] == 'normal':
                    start[walker, i] = np.random.normal(self.parameters[p]['prior']['loc'],
                                                                    self.parameters[p]['prior']['scale'])

        return start


    def get_walkers(self, x0, hess_inv=None):

        if self.sampler_info['initial'] == 'ellipse':
            p0 = self.get_ellipse(x0)
        elif self.sampler_info['initial'] == 'laplace':
            p0 = self.get_laplace(x0, hess_inv)
        elif self.sampler_info['initial'] == 'prior':
            p0 = self.get_prior()
        else:
            print('Use valid initialization...')
        return p0
