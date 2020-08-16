import numpy as np 

def todict(keys, elements):
    return dict(zip(keys, elements.tolist()))

class define_logposterior:

    def __init__(self, params, loglike_fn, logprior_fn):
        self.loglike_fn = loglike_fn
        self.logprior_fn = logprior_fn
        self.labels = [key for key in params['Parameters']]

    def get_logposterior(self, x):
        x = todict(self.labels, x)
        lp = self.logprior_fn(x)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.loglike_fn(x)