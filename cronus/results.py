import numpy as np
import scipy
import h5py
import glob
from tabulate import tabulate


class read_chains:
    r"""
    Class to read ``cronus`` chains. It can be used when either ``zeus`` or ``emcee`` is used as the sampler.

    Parameters
    ----------
    folder : str
        The directory path for run-folder that ``cronus`` created, e.g. chains/run5
    """
    def __init__(self, folder):
        if folder[-1] !=  "/":
            folder += "/"
        self.filenames = glob.glob(folder+"*.h5")

        data = []
        for filename in self.filenames:
            with h5py.File(filename, "r") as hf:
                samples = np.copy(hf['samples'])
                data.append(samples[samples.shape[0]//2:])

        self.nsamples, self.nwalkers, self.ndim = np.shape(data[0])
        self.nchains = len(data)
        
        self.samples = np.empty((self.nsamples, self.nchains * self.nwalkers, self.ndim))
        
        for i in range(self.nchains):
            self.samples[:,i*self.nwalkers:(i+1)*self.nwalkers, :] = data[i]

        self._map = np.load(folder+'MAP.npy')
        self._hessian = np.load(folder+'hessian.npy')

        self._gelmanrubin = np.loadtxt(folder+'GelmanRubin.dat', skiprows=1)[:,1:]

        taus = 0
        self.iter = None
        for i in range(self.nchains):
            tau = np.loadtxt(folder + "IAT_"+str(i)+".dat", skiprows=1)
            taus += tau.T[1:]
            self.iter = tau.T[0]
        taus /= 2
        self._autocorr = taus.T

        with open(folder + "varnames.dat", mode="r") as f:
            varnames = f.read()
        self.varnames  = varnames.split()


    @property
    def MAP(self):
        r"""
        Maximum a Posteriori Estimate.
        """
        return self._map

    @property
    def hessian(self):
        r"""
        Hessian Matrix computed for the Laplace approximation. It can be used as an approximate inverse Covariance Matrix.
        """
        return self._hessian

    @property
    def trace(self):
        r"""
        Trace of chains i.e. Samples flattened to (nsamples, ndim) shape.
        """
        return self.samples.reshape((-1,  self.ndim))

    @property
    def mean(self):
        r"""
        Mean of parameters.
        """
        return np.mean(self.trace, axis=0)

    @property
    def median(self):
        r"""
        Median of parameters.
        """
        return np.median(self.trace, axis=0)

    @property
    def one_sigma(self):
        r"""
        One  sigma constraints on parameters.
        """
        percentiles = [50., 15.86555, 84.13445]
        vals = np.percentile(self.trace, percentiles, axis=0)
        return [-(vals[0] - vals[1]), vals[2] - vals[0]]

    @property
    def two_sigma(self):
        r"""
        Two  sigma constraints on parameters.
        """
        percentiles = [50, 2.2775, 97.7225]
        vals = np.percentile(self.trace, percentiles, axis=0)
        return [-(vals[0] - vals[1]), vals[2] - vals[0]]

    @property
    def three_sigma(self):
        r"""
        Three  sigma constraints on parameters.
        """
        percentiles = [50, 0.135, 99.865]
        vals = np.percentile(self.trace, percentiles, axis=0)
        return [-(vals[0] - vals[1]), vals[2] - vals[0]]

    @property
    def std(self):
        r"""
        Standard deviation of parameters.
        """
        return np.std(self.trace, axis=0)

    @property
    def var(self):
        r"""
        Variance of parameters.
        """
        return np.var(self.trace, axis=0)

    @property
    def cov(self):
        r"""
        Covariance matrix of parameters.
        """
        pass

    @property
    def corr(self):
        r"""
        Correlation matrix of parameters.
        """
        pass


    def GelmanRubin(self, final=True):
        r"""
        Gelman-Rubin Diagnostics.

        Paramaters
        ----------
        final : bool
            Show only the final estimate of Gelman-Rubin R_hat values or show all the estimates during the run.

        Returns
        -------
        (Iterations), R_hat : (Array), Array
        """
        if final:
            return self._gelmanrubin[-1]
        else:
            return self.iter, self._gelmanrubin
        

    def Autocorr(self, final=True):
        r"""
        Integrated Autocorrelation Time (IAT).

        Paramaters
        ----------
        final : bool
            Show only the final estimate of IAT values or show all the estimates during the run.

        Returns
        -------
        (Iterations), IAT : (Array), Array
        """
        if final:
            return self._autocorr[-1]
        else:
            return self.iter, self._autocorr

    @property
    def ESS(self):
        r"""
        Effective Sample Size of chains.
        """
        return self.nsamples * self.nwalkers * self.nchains / self.Autocorr()

    @property
    def Summary(self):
        r"""
        Summary statistics.
        """
        headers = ["Name", "MAP", "mean", "median", "std", "-1 sigma", "+1 sigma", "-2 sigma", "+2 sigma", "IAT", "ESS", "R_hat"]
        data = []
        for i, p in enumerate(self.varnames):
            MAP = str(self.MAP[i])
            mean = str(self.mean[i])
            median = str(self.median[i])
            std = str(self.std[i])
            m1sigma = str(self.one_sigma[0][i])
            p1sigma = str(self.one_sigma[1][i])
            m2sigma = str(self.two_sigma[0][i])
            p2sigma = str(self.two_sigma[1][i])
            IAT = str(self.Autocorr()[i])
            ESS = str(self.ESS[i])
            R_hat = str(self.GelmanRubin()[i])
            data.append([p, MAP, mean, median, std, m1sigma, p1sigma, m2sigma, p2sigma, IAT, ESS, R_hat])
        
        t = tabulate(data, headers=headers, tablefmt='orgtbl')
        return t