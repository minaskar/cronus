import numpy as np
import zeus
from zeus import ChainManager

from scipy.optimize import minimize

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

            start = p0#[cm.get_rank]

            #start0 = np.array([ 1.43563590e+00, -2.79228917e-02,  1.01513786e+01, -2.94868355e+02,
            #                    2.45089136e+03, -3.57808097e+03,  1.85639553e-01, -2.42260305e-01,
            #                   -1.16837933e+03,  1.30581940e+03,  6.21201014e+01,  9.17828705e-01,
            #                   -7.10687852e+01,  7.39951459e+02, -4.96254867e+03,  1.11538809e+04,
            #                    8.99609731e-01, 4.79761397e-01, 7.86391667e+00, 1.50709982e+00,
            #                    1.06320700e+00,  1.00022316e+00])
            #np.random.seed(cm.get_rank)
            #start = 0.01*np.random.randn(self.nwalkers, self.ndim) + start0

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