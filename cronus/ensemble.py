import numpy as np 

class Ensemble:

    def __init__(self, name, nwalkers, ndim, logprob, pool):
        self.name = name
        self.nwalkers = nwalkers
        self.ndim = ndim

        if name == 'zeus':
            import zeus
            self.sampler = zeus.EnsembleSampler(nwalkers, ndim, logprob, pool=pool, verbose=False)
        elif name == 'light':
            import zeus
            self.sampler = zeus.EnsembleSampler(nwalkers, ndim, logprob, pool=pool, verbose=False, light_mode=True)
        elif name == 'emcee':
            import emcee
            #self.state = emcee.State
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, pool=pool)
        elif name == 'demc':
            import emcee
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, moves=[(emcee.moves.DEMove(), 0.9), (emcee.moves.DESnookerMove(), 0.1)], pool=pool)
    

    def sample(self, start, log_prob0, iterations, progress=False, thin=1):

        self.nsteps = iterations
        
        if self.name in ['zeus', 'light']:
            return self.sampler.sample(start=start, log_prob0=log_prob0, iterations=iterations, progress=progress, thin=thin)
        elif self.name in ['emcee', 'demc']:
            #state = self.state(start, log_prob=log_prob0, blobs=None, random_state=None, copy=False)
            return self.sampler.sample(initial_state=start, log_prob0=log_prob0, iterations=iterations, progress=progress, thin=thin)

    
    def reset(self):
        return self.sampler.reset()


    def get_chain(self, *args, **kwargs):
        return self.sampler.get_chain(*args, **kwargs)


    def get_log_prob(self, *args, **kwargs):
        return self.sampler.get_log_prob(*args, **kwargs)


    def get_last(self):
        return self.get_chain()[-1], self.get_log_prob()[-1]

    @property
    def iteration(self):
        return self.sampler.iteration

    @property
    def ncall(self):
        if self.name in ['zeus', 'light']:
            return self.sampler.ncall
        elif self.name in ['emcee', 'demc']:
            return self.nwalkers * self.nsteps