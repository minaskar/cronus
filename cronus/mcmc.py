import numpy as np
import zeus
from zeus import ChainManager


import sys
import time

import emcee

from .save import datasaver
from .diagnostics import diagnose, test_gelmanrubin
from .posterior import define_logposterior
from .start import initialize_walkers

class sampler:

    def __init__(self, params):
        self.params = params
        
        self.name = params['Sampler']['name']
        self.ndim =  params['Sampler']['ndim']
        self.nwalkers =  params['Sampler']['nwalkers']
        self.nsteps =  params['Sampler']['ncheck']
        self.nchains =  params['Sampler']['nchains']
        self.nmin = params['Sampler']['nmin']
        self.nmax = params['Sampler']['nmax']
        self.ncall = params['Sampler']['ncall']

        self.use_act = params['Diagnostics']['Autocorrelation']['use']
        self.dact = params['Diagnostics']['Autocorrelation']['dact']
        self.nact = params['Diagnostics']['Autocorrelation']['nact']

        self.use_gr = params['Diagnostics']['Gelman-Rubin']['use']
        self.epsilon = params['Diagnostics']['Gelman-Rubin']['epsilon']

        self.output = params['Output']

    def run_mcmc(self, loglike_fn, logprior_fn):

        with ChainManager(self.nchains) as cm:

            logpost_fn = define_logposterior(self.params, loglike_fn, logprior_fn).get_logposterior

            if self.name == 'zeus':
                sampler = zeus.sampler(self.nwalkers, self.ndim, logpost_fn, pool=cm.get_pool, verbose=False)
            elif self.name == 'emcee':
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, logpost_fn, pool=cm.get_pool)
            d = datasaver('chain_'+str(cm.get_rank)+'.h5')

            diag = diagnose(tau_epsilon=self.dact, tau_multiple=self.nact)

            ensemble = initialize_walkers(self.params, loglike_fn, logprior_fn)
            start = ensemble.get_walkers()

            if cm.get_rank==0:
                t0 = time.time()
            
            ncall = 0
            cnt = 0
            while True:
                sampler.run_mcmc(start, self.nsteps, progress=False);
                chain = sampler.get_chain()
                start = chain[-1]
                
                d.save(self.output+'chain_'+str(cm.get_rank), chain)
                sampler.reset()

                samples = d.load(self.output+'chain_'+str(cm.get_rank))
                    
                
                diag.add_samples(samples)
                act_converged, tau, delta = diag.test_act()
                mean, var, N = diag.get_gr_details()

                cm.chains_comm.barrier()

                act_converged = cm.gather(act_converged, root=0)
                taus = cm.gather(tau, root=0)
                deltas = cm.gather(delta, root=0)
                means = cm.gather(mean, root=0)
                vars = cm.gather(var, root=0)
                Ns = cm.gather(N, root=0)


                terminate = False
                if cm.get_rank == 0:

                    cnt += self.nsteps
                    if self.name == 'zeus':
                        ncall += sampler.ncall
                    elif self.name == 'emcee':
                        ncall += self.nsteps * self.nwalkers

                    rhat = test_gelmanrubin(means, vars, Ns[0])
                    clock = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
                    print(clock, '| Iter:', cnt, '| ncall:', ncall, '| R-hat:', round(np.max(rhat),4),
                          '| act:', np.max(taus), '| nact:', int(cnt/np.max(taus)), '<', self.nact,
                          '| dact:', np.max(deltas), '<', self.dact, end='\r', flush=True)

                    
                    if cnt > self.nmin:
                        if self.use_gr and self.use_act:
                            if np.all(np.abs(rhat-1.0)<self.epsilon) and np.all(act_converged):
                                print('', flush=True)
                                terminate = True
                        elif self.use_gr and not self.use_act:
                            if np.all(np.abs(rhat-1.0)<self.epsilon):
                                print('', flush=True)
                                terminate = True
                        elif not self.use_gr and self.use_act:
                            if np.all(act_converged):
                                print('', flush=True)
                                terminate = True
                    

                    if cnt>self.nmax:
                        print('', flush=True)
                        terminate = True

                    if ncall>self.ncall:
                        print('', flush=True)
                        terminate = True
                    
                terminate = cm.bcast(terminate, root=0)
                if terminate:
                    break