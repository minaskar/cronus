import h5py
import numpy as np

class datasaver:

    def __init__(self, filename):
        self.filename = filename
        self.initialised = False

    def save(self, name, data):

        nsamples, nwalkers, ndim = np.shape(data)

        if not self.initialised:
            with h5py.File(self.filename, 'w') as hf:
                hf.create_dataset(name, data=data, compression="gzip", chunks=True, maxshape=(None, nwalkers, ndim)) 
            self.initialised  = True
        else:
            with h5py.File(self.filename, 'a') as hf:
                hf[name].resize((hf[name].shape[0] + nsamples), axis = 0)
                hf[name][-nsamples:] = data
        
    def load(self, name):

        with h5py.File(self.filename, "r") as hf:
            data = np.copy(hf[name])
        return data