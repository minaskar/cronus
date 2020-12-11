import numpy as np

from zeus import AutoCorrTime


def GelmanRubin(chains):
    '''
    Gelman-Rubin convergence statistic.
    '''

    chains = np.array(chains)

    # Number and length of chains
    M, N, _ = np.shape(chains)

    # Within chain variance
    W = 0.0
    for chain in chains:
        W += np.var(chain, axis=0)
    W /= M

    # Means of chains
    means = []
    for chain in chains:
        means.append(np.mean(chain, axis=0))
    means = np.array(means)

    # Between chain variance
    B = N * np.var(means, ddof=1, axis=0)

    # Weighted variance
    Var_hat = (1.0 - 1.0 / N) * W + B / N

    # Return R_hat statistic
    R_hat = np.sqrt(Var_hat / W)

    return R_hat


class diagnose:

    def __init__(self, tau_epsilon=0.01, tau_multiple=100, thin=1):
        self.taus = [np.inf]
        self.tau_epsilon = tau_epsilon
        self.tau_multiple = tau_multiple
        self.samples = None
        self.ndim = 1
        self.thin = thin
        self.acts = []


    def add_samples(self, samples):
        self.samples = samples
        self.nsamples = np.shape(samples)[0]
        self.ndim = np.shape(samples)[2]


    def test_act(self):
        old_tau = self.taus[-1]
        act = AutoCorrTime(self.samples[self.nsamples//2:]) * self.thin
        tau = np.mean(act)
        self.taus.append(tau)
        self.acts.append(act)

        converged = np.all(tau * self.tau_multiple / self.thin < self.nsamples)
        delta = np.abs(tau-old_tau) / tau
        #delta = (tau - old_tau) / tau
        converged &= np.all(delta < self.tau_epsilon)
        return converged, round(tau,1), round(delta,3)

    
    def get_gr_details(self):
        
        chain0 = self.samples[self.nsamples//2:3*self.nsamples//4].reshape((-1,self.ndim))
        mean0 = np.mean(chain0, axis=0)
        var0 = np.var(chain0, axis=0)

        chain1 = self.samples[3*self.nsamples//4:].reshape((-1,self.ndim))
        mean1 = np.mean(chain1, axis=0)
        var1 = np.var(chain1, axis=0)
        return [mean0, mean1], [var0, var1], np.shape(chain0)[0]    


def test_gelmanrubin(means, vars, N):

    _means = [item for sublist in means for item in sublist]
    _vars = [item for sublist in vars for item in sublist]

    # Between chain variance
    B = N * np.var(_means, ddof=1, axis=0)

    # Within chain variance
    W = np.mean(_vars)

    # Weighted variance
    Var_hat = (1.0 - 1.0 / N) * W + B / N

    # Return R_hat statistic
    R_hat = np.sqrt(Var_hat / W)
    
    return R_hat
