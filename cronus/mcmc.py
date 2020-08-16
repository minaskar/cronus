import numpy as np
import zeus
from zeus import ChainManager

import sys

import emcee

from .save import datasaver
from .diagnostics import diagnose, test_gelmanrubin

class sampler:

    def __init__(self, params):
        sampler_info = params['Sampler']
        self.name = sampler_info['name']
        self.ndim =  sampler_info['ndim']
        self.nwalkers =  sampler_info['nwalkers']
        self.nsteps =  sampler_info['nsteps']
        self.nchains =  sampler_info['nchains']

    def run_mcmc(self, logpost_fn, p0):

        with ChainManager(self.nchains) as cm:

            if self.name == 'zeus':
                sampler = zeus.sampler(self.nwalkers, self.ndim, logpost_fn, pool=cm.get_pool, verbose=False)
            elif self.name == 'emcee':
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, logpost_fn, pool=cm.get_pool)
            d = datasaver('chain_'+str(cm.get_rank)+'.h5')

            diag = diagnose(tau_epsilon=0.05, tau_multiple=10)

            start = p0[cm.get_rank]

            cnt = 0

            while True:
                sampler.run_mcmc(start, self.nsteps, progress=False);
                chain = sampler.get_chain()
                start = chain[-1]
                
                d.save('chain_'+str(cm.get_rank), chain)
                sampler.reset()

                samples = d.load('chain_'+str(cm.get_rank))
                
                cnt += self.nsteps
                
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
                    rhat = test_gelmanrubin(means, vars, Ns[0])
                
                    print('Iter:', cnt, '| Max R-hat:', round(np.max(rhat),4), '| ACT:', taus, 'Dtau:', deltas, end='\r')
                    sys.stdout.flush()

                    if np.all(np.abs(rhat-1.0)<0.015) and np.all(act_converged):
                        print('')
                        sys.stdout.flush()
                        print('Done')
                        sys.stdout.flush()
                        terminate = True
                    
                terminate = cm.bcast(terminate, root=0)
                if terminate:
                    break