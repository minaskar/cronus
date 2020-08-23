import numpy as np 

def todict(keys, elements):
    return dict(zip(keys, elements.tolist()))

class define_logposterior:

    def __init__(self, params, loglike_fn, logprior_fn, beta=1.0):
        self.loglike_fn = loglike_fn
        self.logprior_fn = logprior_fn
        self.labels = [key for key in params['Parameters']]
        self.beta = beta

    def get_logposterior(self, x):
        x = todict(self.labels, x)
        lp = self.logprior_fn(x)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike_fn(x) * self.beta

    def get_neglogposterior(self, x):
        x = todict(self.labels, x)
        lp = self.logprior_fn(x)
        if not np.isfinite(lp):
            return np.inf
        return -(lp + self.loglike_fn(x) * self.beta)

    def get_logprior(self, x):
        x = todict(self.labels, x)
        return self.logprior_fn(x)