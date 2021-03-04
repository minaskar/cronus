import numpy as np
import time
from zeus import AutoCorrTime
import h5py

class Diagnostics:

    def __init__(self,
                 dataloader=None,
                 tau_epsilon=0.01,
                 tau_multiple=100,
                 epsilon=0.03,
                 use_act=True,
                 use_gr=True,
                 miniter=100,
                 maxiter=np.inf,
                 maxcall=np.inf,
                 thin=1,
                 size=1,
                 continue_mcmc=False):

        self.dataloader = dataloader
        self.ndim = dataloader.ndim
        self.output = dataloader.output

        self.tau_epsilon = tau_epsilon
        self.tau_multiple = tau_multiple
        self.epsilon = epsilon 
        self.use_act = use_act
        self.use_gr = use_gr

        self.miniter = miniter
        self.maxiter = maxiter
        self.maxcall = maxcall

        self.thin = thin 
        self.size = size

        self.convergence = False
        
        if continue_mcmc:
            with h5py.File(self.output+'diagnostics.h5', "r") as hf:
                self.Rhats = list(np.copy(hf['Rhat']))
                self.taus = list(np.copy(hf['tau']))
                self.deltas = list(np.copy(hf['delta']))
                self.initialised = True
        else:
            self.Rhats = [2.0]
            self.taus = [1.0]
            self.deltas = [1.0]
            self.initialised = False

        self.Rhat = self.Rhats[-1]
        self.tau = self.taus[-1]
        self.delta = self.deltas[-1]

    
    def diagnose(self, cnt, ncall):

        means_all = []
        vars_all = []
        act_all = np.empty((self.size, self.ndim))

        for i in range(self.size):
            
            samples = self.dataloader.get_samples(rank=i)
            nsamples, _, _ = samples.shape
            act_all[i] = self.estimate_act(samples)

            means_1chain, vars_1chain, N = self.estimate_gr_split(samples)
            means_all.append(means_1chain)
            vars_all.append(vars_1chain)
        
        Rhat_all = self.estimate_Rhat(means_all, vars_all, N)
        Rhat_max = np.max(Rhat_all)
        self.Rhats.append(Rhat_max)
        self.Rhat = Rhat_max

        tau_mean = np.mean(act_all)
        self.taus.append(tau_mean)
        self.tau = tau_mean

        self.convergence, self.delta = self.test_conv(nsamples)
        self.deltas.append(self.delta)

        if cnt > self.miniter:
            if cnt >= self.maxiter or ncall >= self.maxcall:
                self.convergence = True 

        if self.initialised:
            with h5py.File(self.output+'diagnostics.h5', 'a') as hf:
                for i in range(self.size):
                    hf['act'+str(i)].resize((hf['act'+str(i)].shape[0] + 1), axis = 0)
                    hf['act'+str(i)][-1] = act_all[i]
                hf['gr'].resize((hf['gr'].shape[0] + 1), axis = 0)
                hf['gr'][-1] = Rhat_all
                hf['Rhat'].resize((hf['Rhat'].shape[0] + 1), axis = 0)
                hf['Rhat'][-1] = Rhat_max
                hf['tau'].resize((hf['tau'].shape[0] + 1), axis = 0)
                hf['tau'][-1] = tau_mean
                hf['delta'].resize((hf['delta'].shape[0] + 1), axis = 0)
                hf['delta'][-1] = self.delta
        else:
            with h5py.File(self.output+'diagnostics.h5', 'w') as hf:
                for i in range(self.size):
                    hf.create_dataset('act'+str(i), data=act_all[i].reshape(1,-1), compression="gzip", chunks=True, maxshape=(None,self.ndim))
                hf.create_dataset('gr', data=Rhat_all.reshape(1,-1), compression="gzip", chunks=True, maxshape=(None,self.ndim)) 
                hf.create_dataset('Rhat', data=Rhat_max.reshape(1), compression="gzip", chunks=True, maxshape=(None,)) 
                hf.create_dataset('tau', data=tau_mean.reshape(1), compression="gzip", chunks=True, maxshape=(None,))
                hf.create_dataset('delta', data=self.delta.reshape(1), compression="gzip", chunks=True, maxshape=(None,))
            self.initialised  = True

        return self.convergence

    
    def estimate_act(self, samples):
        nsamples = np.shape(samples)[0]
        return AutoCorrTime(samples[nsamples//2:]) * self.thin


    def estimate_gr_split(self, samples):
        nsamples, _, ndim = np.shape(samples)

        chain0 = samples[nsamples//2:3*nsamples//4].reshape((-1,ndim))
        mean0 = np.mean(chain0, axis=0)
        var0 = np.var(chain0, axis=0)

        chain1 = samples[3*nsamples//4:].reshape((-1,ndim))
        mean1 = np.mean(chain1, axis=0)
        var1 = np.var(chain1, axis=0)

        return [mean0, mean1], [var0, var1], np.shape(chain0)[0]    


    def test_conv(self, nsamples):
        if self.use_gr:
            gr_conv = self.test_gr()
        else:
            gr_conv = True 

        if self.use_act:
            act_conv, dact = self.test_act(nsamples)
        else:
            act_conv = True
            dact = 1.0

        return gr_conv & act_conv, dact
        

    
    def test_act(self, nsamples):
        nact_conv = self.test_nact(nsamples)
        dact_conv, dact = self.test_dact()
        return nact_conv & dact_conv, dact


    def test_nact(self, nsamples):
        if self.taus[-1] * self.tau_multiple / self.thin < nsamples:
            return True
        else:
            return False


    def test_dact(self):
        delta = np.abs(self.taus[-1]-self.taus[-2]) / self.taus[-1]
        return delta <= self.tau_epsilon, delta


    def test_gr(self):
        if np.abs(self.Rhats[-1] - 1.0) <= self.epsilon:
            return True
        else:
            return False


    def estimate_Rhat(self, means, vars, N):

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