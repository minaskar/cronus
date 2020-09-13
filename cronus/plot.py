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


def triangleplot(results, varnames=None, thin=10, height=2.5,
                 scatterplot=True, histplot=True, kdeplot=True,
                 s=5, scatter_color=".15", bins=30, pthresh=0.15, cmap="mako",
                 kde_color="w", linewidths=2, linewidths_diag=2,
                 savefig=False, filename='triangleplot.png', dpi=200):

    data = results.trace[::thin]
    df = pd.DataFrame(data=data, index=[i for i in range(data.shape[0])], columns=results.varnames)

    if varnames is not None:
        df = df[varnames]

    g = sns.PairGrid(df, diag_sharey=True, corner=True, despine=False, height=height)
    if scatterplot:
        g.map_lower(sns.scatterplot, s=s, color=scatter_color)
    if histplot:
        g.map_lower(sns.histplot, bins=bins, pthresh=pthresh, cmap=cmap)
    if kdeplot:
        g.map_lower(sns.kdeplot, levels=[0.3935, 0.8647], color=kde_color, linewidths=linewidths)
    g.map_diag(sns.kdeplot, lw=linewidths_diag)
    g.tight_layout()
    if savefig:
        g.savefig(filename, dpi=dpi)
