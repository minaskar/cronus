import numpy as np
from zeus import ChainManager

import sys
import time

from .save import datasaver
from .diagnostics import diagnose, test_gelmanrubin

from .bayes import Distribution
from .likelihood import import_loglikelihood
from .start import initialize_walkers
from .optimize import find_MAP

class sampler:

    def __init__(self, params):
        self.params = params
        
        # Sampler
        self.name = params['Sampler']['name']
        self.ndim =  params['Sampler']['ndim']
        self.nwalkers =  params['Sampler']['nwalkers']
        self.nsteps =  params['Sampler']['ncheck']
        self.nchains =  params['Sampler']['nchains']
        self.miniter = params['Sampler']['miniter']
        self.maxiter = params['Sampler']['maxiter']
        self.maxcall = params['Sampler']['maxcall']
        self.thin = params['Sampler']['thin']

        # Diagnostics
        self.use_act = params['Diagnostics']['Autocorrelation']['use']
        self.dact = params['Diagnostics']['Autocorrelation']['dact']
        self.nact = params['Diagnostics']['Autocorrelation']['nact']

        self.use_gr = params['Diagnostics']['Gelman-Rubin']['use']
        self.epsilon = params['Diagnostics']['Gelman-Rubin']['epsilon']

        # Output
        self.output = params['Output']

    
    def run(self, loglike_fn):
        if self.name in ['zeus', 'emcee']:
            self.run_mcmc(loglike_fn)
        elif self.name  in ['dynesty']:
            self.run_nested(loglike_fn)


    def run_mcmc(self, loglike_fn):

        if self.name == 'zeus':
            import zeus
        elif self.name == 'emcee':
            import emcee

        with ChainManager(self.nchains) as cm:

            rank = cm.get_rank

            # Define Log Posterior
            distribution = Distribution(self.params, loglike_fn)
            logpost_fn = distribution.get_logposterior
            
            # Initialize Sampler
            if self.name == 'zeus':
                sampler = zeus.sampler(self.nwalkers, distribution.nfree, logpost_fn, pool=cm.get_pool, verbose=False)
            elif self.name == 'emcee':
                sampler = emcee.EnsembleSampler(self.nwalkers, distribution.nfree, logpost_fn, pool=cm.get_pool)

            # Initialize Datasaver
            d = datasaver(self.output+'chain_'+str(cm.get_rank)+'.h5')

            # Initialize Diagnostics
            diag = diagnose(tau_epsilon=self.dact, tau_multiple=self.nact, thin=self.thin)

            # Initialize Walkers
            ensemble = initialize_walkers(self.params, distribution)
            x0, f0, h0 = find_MAP(self.params, distribution, ensemble.bounds)
            x0s = cm.allgather(x0)
            f0s = cm.allgather(f0)
            h0s = cm.allgather(h0)
            start = ensemble.get_walkers(x0s[np.argmin(f0s)], h0s[np.argmin(f0s)])

            # Start Sampling Loop
            if rank==0:
                t0 = time.time()
            
            ncall = 0
            cnt = 0
            while True:
                sampler.run_mcmc(start, self.nsteps, progress=False, thin=self.thin);
                chain = sampler.get_chain()
                logps = sampler.get_log_prob()
                start = chain[-1]
                d.save(chain, logps)
                sampler.reset()

                samples = d.load('samples')
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
                
                if rank == 0:
                    terminate = False

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

                    
                    if cnt > self.miniter:
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
                    

                    if cnt>self.maxiter:
                        print('', flush=True)
                        terminate = True

                    if ncall>self.maxcall:
                        print('', flush=True)
                        terminate = True

                else:
                    terminate = False
                    
                cm.chains_comm.barrier()
                terminate = cm.bcast(terminate, root=0)
                cm.chains_comm.barrier()
                if terminate:
                    break

        
    def run_nested(self, loglike_fn):

        import dynesty

        with ChainManager(1) as cm:

            # Define Log Posterior
            distribution = Distribution(self.params, loglike_fn)
            loglike_fn = distribution.get_loglikelihood
            ptform = distribution.get_prior_transform

            sampler = dynesty.DynamicNestedSampler(loglike_fn, ptform, distribution.nfree, pool=cm.get_pool, use_pool={'prior_transform': False})
            sampler.run_nested(wt_kwargs={'pfrac': 1.0})

            res = sampler.results