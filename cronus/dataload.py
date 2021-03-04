import numpy as np
import h5py

from zeus import AutoCorrTime

class DataLoader:

    def __init__(self, output, ndim, size=1, continue_mcmc=False):
        self.size = size
        self.output = output
        self.ndim = ndim
        if continue_mcmc:
            self.initialised = True
        else:
            self.initialised = False

    def save(self, samples, logps):

        if self.initialised:
            with h5py.File(self.output+'data.h5', 'a') as hf:
                for i in range(self.size):
                    hf['samples'+str(i)].resize((hf['samples'+str(i)].shape[0] + samples[i].shape[0]), axis = 0)
                    hf['samples'+str(i)][-samples[i].shape[0]:] = samples[i]
                    hf['logprob'+str(i)].resize((hf['logprob'+str(i)].shape[0] + logps[i].shape[0]), axis = 0)
                    hf['logprob'+str(i)][-logps[i].shape[0]:] = logps[i]
        else:
            with h5py.File(self.output+'data.h5', 'w') as hf:
                for i in range(self.size):
                    hf.create_dataset('samples'+str(i), data=samples[i], compression="gzip", chunks=True, maxshape=(None,)+samples[i].shape[1:])
                    hf.create_dataset('logprob'+str(i), data=logps[i], compression="gzip", chunks=True, maxshape=(None,)+logps[i].shape[1:]) 
            self.initialised  = True


    def load(self, name, rank=0):
        with h5py.File(self.output+'data.h5', "r") as hf:
            data = np.copy(hf[name+str(rank)])
        return data


    def get_samples(self, rank=0):
        return self.load('samples', rank=rank)


    def get_logps(self, rank=0):
        return self.load('logprob', rank=rank)


    def get_last_sample(self, rank=0):
        return self.load('samples', rank=rank)[-1]


    def get_last_logps(self, rank=0):
        return self.load('logprob', rank=rank)[-1]