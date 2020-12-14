import numpy as np
from zeus import ChainManager

import os
import sys
import time
from ruamel.yaml import YAML

from .save import datasaver
from .diagnostics import diagnose, test_gelmanrubin
from .bayes import Distribution
from .results import read_chains
from .helpers import damp_paramfile, initialise_sampler, create_gelmanrubin, create_varnames, create_IAT, get_starting_positions
from .helpers import append_IAT, append_GR, print_status, save_results


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
        self.output = params['Output'] + "/"

        #self.comm = comm

    
    def run(self, loglike_fn):
        if self.name in ['zeus', 'emcee']:
            self.run_mcmc(loglike_fn)
        elif self.name in ['dynesty']:
            self.run_nested(loglike_fn)


    def run_mcmc(self, loglike_fn):

        with ChainManager(self.nchains) as cm:

            rank = cm.get_rank

            # Create run folder
            if rank == 0:
                self.output = damp_paramfile(self.output, self.params)
            self.output = cm.bcast(self.output, root=0)

            # Define Log Posterior
            distribution = Distribution(self.params, loglike_fn)
            logpost_fn = distribution.get_logposterior
            
            # Initialize Sampler
            sampler = initialise_sampler(self.name, self.nwalkers, distribution.nfree, logpost_fn, pool=cm.get_pool)
            
            # Initialize Datasaver
            d = datasaver(self.output+'chain_'+str(cm.get_rank)+'.h5')

            # Initialize Diagnostics
            diag = diagnose(tau_epsilon=self.dact, tau_multiple=self.nact, thin=self.thin)

            # Initialize Walkers
            if rank==0:
                print('Initializing walker positions...', end='\r', flush=True)
            start, MAP, hessian = get_starting_positions(self.params, distribution, cm)

            if rank==0:
                print('Walkers initialised...', end='\r', flush=True)

                np.save(self.output+'MAP.npy', MAP)
                np.save(self.output+'hessian.npy', hessian)
                create_gelmanrubin(self.output, distribution.free_labels)
                create_varnames(self.output, distribution.free_labels)

            create_IAT(self.output, distribution.free_labels, rank)
            
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

                cnt += self.nsteps

                samples = d.load('samples')
                diag.add_samples(samples)
                act_converged, tau, delta = diag.test_act()
                act = diag.acts[-1]
                append_IAT(self.output, cnt, act, rank)
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

                    if self.name == 'zeus':
                        ncall += sampler.ncall
                    elif self.name == 'emcee':
                        ncall += self.nsteps * self.nwalkers

                    rhat = test_gelmanrubin(means, vars, Ns[0])
                    append_GR(self.output, cnt, rhat)

                    print_status(t0, cnt, ncall, rhat, taus, self.nact, deltas, self.dact)

                    
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
            
            if rank == 0:
                results = read_chains(self.output)
                print(results.Summary, flush=True)
                save_results(self.output, results)

        
    def run_nested(self, loglike_fn):

        import dynesty

        with ChainManager(1) as cm:

            rank = cm.get_rank

            # Create run folder
            if rank == 0:
                nrun = 1
                while True:
                    path = self.output + "run" + str(nrun) + "/"
                    if os.path.isdir(path):
                        nrun += 1
                    else:
                        self.output = path
                        break
                os.makedirs(path)
                # Dump full parameter file
                yaml = YAML()
                with open(self.output + "para.yaml", mode='w') as f:
                    yaml.dump(self.params, f)
            self.output = cm.bcast(self.output, root=0)

            # Define Log Posterior
            distribution = Distribution(self.params, loglike_fn)
            loglike_fn = distribution.get_loglikelihood
            ptform = distribution.get_prior_transform

            sampler = dynesty.DynamicNestedSampler(loglike_fn, ptform, distribution.nfree, pool=cm.get_pool, use_pool={'prior_transform': False})
            sampler.run_nested(wt_kwargs={'pfrac': 1.0})

            res = sampler.results