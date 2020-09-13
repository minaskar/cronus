import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


def traceplot(results, varnames=None, width=12, height=1.75, fontsize=14, show_mean=True, savefig=False, filename="traceplot.png", dpi=200):
    if varnames is not None:
        npar = len(varnames)
    else:
        npar = results.ndim
        varnames = results.varnames

    fig = plt.figure(figsize=(width,height*npar))
    plt.rcParams.update({'font.size': fontsize})
    for j in range(results.ndim):
        if results.varnames[j] in varnames:
            for i in range(results.nwalkers*results.nchains):
                ax = fig.add_subplot(npar,2,j*2+1)
                sns.kdeplot(results.samples[:,i,j])
                if show_mean:
                    ax.axvline(x=results.mean[j], c='k', ls='--')
                plt.xlabel(results.varnames[j])
                plt.ylabel('')
            if show_mean:
                sns.kdeplot(results.samples[:,:,j].reshape(-1), c='k', ls='--')
            plt.ylabel('')
            ax = fig.add_subplot(npar,2,j*2+2)
            ax.plot(results.samples[:,:,j], alpha=0.6)
            if show_mean:
                ax.axhline(y=results.mean[j], c='k', ls='--', alpha=0.6)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            plt.ylabel(results.varnames[j])
    plt.tight_layout()
    if savefig:
        plt.savefig(filename, dpi=dpi)
    plt.show()


def cornerplot(samples,
               labels=None,
               weights=None,
               levels=None,
               quantiles=[0.025, 0.5, 0.975],
               color=None,
               alpha=0.5,
               linewidth=1.5,
               fill=True,
               show_titles=True,
               title_fmt='.2f',
               cut=3,
               fig=None,
               size=(10,10)):

    nsamples, ndim = samples.shape

    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    if levels is None:
        levels = list(1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2))
    levels.append(1.0)
    
    if color is None:
        color = "tab:blue"

    idxs = np.arange(ndim**2).reshape(ndim, ndim)
    tril = np.tril_indices(ndim)
    triu = np.triu_indices(ndim)
    lower = list(set(idxs[tril])-set(idxs[triu]))
    upper = list(set(idxs[triu])-set(idxs[tril]))
    
    if fig is None:
        figure, axes = plt.subplots(ndim, ndim, figsize=size, sharex=True)
    else:
        figure = fig[0]
        axes = fig[1]

    for idx, ax in enumerate(axes.flat):

        i = idx // ndim
        j = idx % ndim
        
        if idx in lower:
            if fill:
                ax = sns.kdeplot(x=samples[:,j], y=samples[:,i], weights=weights,
                                fill=True, color=color,
                                clip=None, cut=cut,
                                thresh=levels[0], levels=levels,
                                ax=ax, alpha=alpha, linewidth=0.0,
                                )
            ax = sns.kdeplot(x=samples[:,j], y=samples[:,i], weights=weights,
                             fill=False, color=color,
                             clip=None, cut=cut,
                             thresh=levels[0], levels=levels,
                             ax=ax, alpha=alpha, linewidth=linewidth,
                             )

            if j == 0:
                ax.set_ylabel(labels[i])
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                
            else:
                ax.yaxis.set_ticklabels([])

            if i == ndim - 1:
                ax.set_xlabel(labels[j])
            
        elif idx in upper:
            ax.set_axis_off()
        else:
            if fill:
                ax = sns.kdeplot(x=samples[:,j],
                                fill=True, color=color, weights=weights,
                                clip=None, cut=cut,
                                ax=ax, linewidth=0.0, alpha=alpha,
                                )
            ax = sns.kdeplot(x=samples[:,j],
                             fill=None, color=color, weights=weights,
                             clip=None, cut=cut,
                             ax=ax, linewidth=linewidth, alpha=alpha,
                             )

            if i == ndim - 1:
                ax.set_xlabel(labels[j])

            if show_titles:
                ql, qm, qh = _quantile(samples[:,i], quantiles, weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title)
            

            ax.set_ylabel("")
            ax.set_yticks([])

        [l.set_rotation(45) for l in ax.get_xticklabels()]
        [l.set_rotation(45) for l in ax.get_yticklabels()]

        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        
    
    figure.subplots_adjust(top=0.95, right=0.95, wspace=.05, hspace=.05)

    return figure, axes


def _quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.
    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.
    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.
    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles