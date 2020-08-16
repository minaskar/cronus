import numpy as np 

class import_logprior:

    def __init__(self, params):
        self.parameters = params['Parameters']

    def get_logprior(self, x):
        logp = 0.0
        for p in x:
            if self.parameters[p]['prior']['type'] == 'uniform':
                if x[p] < self.parameters[p]['prior']['min'] or x[p] > self.parameters[p]['prior']['max']:
                    return -np.inf
            elif self.parameters[p]['prior']['type'] == 'normal':
                diff = x[p]-self.parameters[p]['prior']['loc']
                logp += -0.5*(diff)**2.0 /self.parameters[p]['prior']['scale']

        return logp