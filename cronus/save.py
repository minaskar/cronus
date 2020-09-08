import h5py
import numpy as np

class datasaver:

    def __init__(self, filename):
        self.filename = filename
        self.initialised = False


    def save(self, samples, logps):

        if not self.initialised:
            with h5py.File(self.filename, 'w') as hf:
                hf.create_dataset('samples', data=samples, compression="gzip", chunks=True, maxshape=(None,)+samples.shape[1:])
                hf.create_dataset('logprob', data=logps, compression="gzip", chunks=True, maxshape=(None,)+logps.shape[1:]) 
            self.initialised  = True
        else:
            with h5py.File(self.filename, 'a') as hf:
                hf['samples'].resize((hf['samples'].shape[0] + samples.shape[0]), axis = 0)
                hf['samples'][-samples.shape[0]:] = samples

                hf['logprob'].resize((hf['logprob'].shape[0] + logps.shape[0]), axis = 0)
                hf['logprob'][-logps.shape[0]:] = logps
        

    def load(self, name):

        with h5py.File(self.filename, "r") as hf:
            data = np.copy(hf[name])
        return data