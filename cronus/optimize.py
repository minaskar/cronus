import numpy as np
from iminuit import minimize
from .posterior import define_logposterior

def find_MAP(params, loglike_fn, logprior_fn, bounds, ntemp=20, return_hess_inv=False):

    x0 = np.empty(len(bounds))
    for i in range(len(bounds)):
        low = bounds[i][0]
        high = bounds[i][1]
        x0[i] = low + np.random.rand() * (high - low)

    idxs = np.arange(1, ntemp + 1)
    betas = 2**(0.5*(idxs-idxs[-1]))
    

    for beta in betas:

        func = define_logposterior(params, loglike_fn, logprior_fn, beta=beta).get_neglogposterior

        cnt = 0
        while True:
            result = minimize(func, x0, options={'maxfev' : 1000}, bounds=bounds)
            x0 = result.x
            cnt += result.nfev
            if result.message == 'Optimization terminated successfully.' or cnt > 20000:
                break

    if return_hess_inv:
        return result.x, result.hess_inv 
    else:
        return result.x
