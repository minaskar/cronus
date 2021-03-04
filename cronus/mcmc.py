import numpy as np
from zeus import ChainManager

import os
import sys
import time
from ruamel.yaml import YAML

from .bayes import Distribution
from .dataload import DataLoader 
from .diagnose import Diagnostics
from .ensemble import Ensemble
from .results import read_chains
from .helpers import damp_paramfile, get_starting_positions
from .helpers import save_results, progress_bar, save_config


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
        self.output = params['Output']['directory'] #+ "/"
        
        #self.comm = comm

    
    def run(self, loglike_fn, continue_mcmc=False):
        if self.name in ['zeus', 'emcee', 'light', 'demc']:
            self.run_mcmc(loglike_fn, continue_mcmc)
        elif self.name in ['dynesty']:
            self.run_nested(loglike_fn)


    def run_mcmc(self, loglike_fn, continue_mcmc):

        with ChainManager(self.nchains) as cm:

            rank = cm.get_rank

            # Define Log Posterior
            distribution = Distribution(self.params, loglike_fn)

            # Initialise Dataloader and Diagnostics
            if not continue_mcmc:
                self.output = damp_paramfile(self.output, self.params, rank)

            dataloader = DataLoader(output=self.output,
                                    ndim=distribution.nfree,
                                    size=self.nchains,
                                    continue_mcmc=continue_mcmc)

            diagnostics = Diagnostics(dataloader=dataloader,
                                          tau_epsilon=self.dact,
                                          tau_multiple=self.nact,
                                          epsilon=self.epsilon,
                                          use_act=self.use_act,
                                          use_gr=self.use_gr,
                                          miniter=self.miniter,
                                          maxiter=self.maxiter,
                                          maxcall=self.maxcall,
                                          thin=self.thin,
                                          size=self.nchains,
                                          continue_mcmc=continue_mcmc)

            # Initialize Sampler
            sampler = Ensemble(name=self.name,
                               nwalkers=self.nwalkers, 
                               ndim=distribution.nfree,
                               logprob=distribution.get_logposterior,
                               pool=cm.get_pool)

            # Initialize Walkers
            if rank==0:
                print('Initializing walker positions...', end='\r', flush=True)
            if continue_mcmc:
                start = dataloader.get_last_sample(rank=rank)
                log_prob0 = dataloader.get_last_logps(rank=rank)
            else:
                start, MAP, hessian = get_starting_positions(self.params, distribution, cm)
                log_prob0 = None
                save_config(self.output, start, MAP, hessian, distribution.free_labels, cm, rank)
 
            # Start Sampling Loop
            ncall = 0
            cnt = 0
            pbar = progress_bar(rank=rank)
            terminate = False
            while True:
                for _ in sampler.sample(start, log_prob0=log_prob0, iterations=self.nsteps, progress=False, thin=self.thin):
                    pbar.update(cnt+sampler.iteration, ncall+sampler.ncall, rhat=diagnostics.Rhat, tau=diagnostics.tau, dact=diagnostics.delta, nact=self.nact, rank=rank)
                    
                start, log_prob0 = sampler.get_last()

                chains_all = cm.gather(sampler.get_chain(), root=0)
                logps_all = cm.gather(sampler.get_log_prob(), root=0)

                ncall += sampler.ncall
                cnt += sampler.iteration

                if rank==0:
                    dataloader.save(chains_all, logps_all)
                    if diagnostics.diagnose(cnt, ncall):
                        terminate = True
                terminate = cm.bcast(terminate, root=0)
                
                sampler.reset()
                
                if terminate:
                    break

            pbar.close(rank=rank)

        
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