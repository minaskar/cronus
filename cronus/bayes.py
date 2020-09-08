import numpy as np
from scipy.stats import norm
import importlib.util
import sys
import inspect


def todict(keys, elements):
    return dict(zip(keys, elements.tolist()))


class Distribution:

    def __init__(self, params, loglike_fn):
        self.params = params

        self.parameters = params['Parameters']
        self.labels = [key for key in params['Parameters']]
        self.ndim =  params['Sampler']['ndim']

        self.dictionary = params['Likelihood']['dictionary']
        self.fixed_mask, self.fixed_values, self.nfixed = self._get_fixed_parameters()
        self.free_mask = self.fixed_mask == False
        self.nfree = self.ndim - self.nfixed
        self.free_labels = list(np.array(self.labels)[self.free_mask])

        self.loglike_fn = loglike_fn
        self.beta = 1.0


    def _get_fixed_parameters(self):

        fixed_mask = np.empty(self.ndim, dtype=bool)
        fixed_values = np.empty(self.ndim)

        for i, p in enumerate(self.parameters):
            if 'prior' in self.parameters[p]:
                fixed_mask[i] = False
                fixed_values[i] = np.nan
            elif 'fixed' in self.parameters[p]:
                fixed_mask[i] = True
                fixed_values[i] = self.parameters[p]['fixed']

        return fixed_mask, fixed_values[fixed_mask], len(fixed_values[fixed_mask])


    def _get_logprior(self, x):
        logp = 0.0
        for p in x:
            if self.parameters[p]['prior']['type'] == 'uniform':
                if x[p] < self.parameters[p]['prior']['min'] or x[p] > self.parameters[p]['prior']['max']:
                    return -np.inf
            elif self.parameters[p]['prior']['type'] == 'normal':
                diff = x[p]-self.parameters[p]['prior']['loc']
                logp += -0.5*(diff/self.parameters[p]['prior']['scale'])**2.0 

        return logp

    
    def get_logprior(self, x):
        x = todict(self.free_labels, x)
        return self._get_logprior(x)

    
    def _free_to_full(self, x):
        x_full = np.empty(self.ndim)
        x_full[self.free_mask] = x
        x_full[self.fixed_mask] = self.fixed_values
        return x_full


    def get_logposterior(self, x):
        lp = self.get_logprior(x)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.get_loglikelihood(x) * self.beta

    
    def get_loglikelihood(self, x):
        x = self._free_to_full(x)
        if self.dictionary:
            x = todict(self.labels, x)
        return self.loglike_fn(x)


    def get_neglogposterior(self, x):
        return -self.get_logposterior(x)

    
    def set_beta(self, beta=1.0):
        self.beta = beta


    def get_prior_transform(self, u):

        ptform = np.empty_like(u)
        t = norm.ppf(u)

        for i, p in enumerate(self.free_labels):
            if self.parameters[p]['prior']['type'] == 'uniform':
                low = self.parameters[p]['prior']['min']
                high = self.parameters[p]['prior']['max']
                ptform[i] = low + u[i] * (high - low)
            elif self.parameters[p]['prior']['type'] == 'normal':
                scale = self.parameters[p]['prior']['scale']
                ptform[i] = t[i] * scale * scale + self.parameters[p]['prior']['loc']

        return ptform