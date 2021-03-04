import numpy as np

def get_default(params):

    # General

    if 'Likelihood' not in params:
        raise KeyError('Please provide information about the Likelihood.')

    if 'Parameters' not in params:
        raise KeyError('Please provide information about the Parameters.')

    if 'Sampler' not in params:
        params['Sampler'] = {}

    if 'Diagnostics' not in params:
        params['Diagnostics'] = {}

    if 'Output' not in params:
        params['Output'] = {}

    # Likelihood

    if 'path' not in params['Likelihood']:
        raise KeyError('Please provide the directory path to the file containing the log_prob function.')

    if 'dictionary' not in params['Likelihood']:
        params['Likelihood']['dictionary'] = False

    # Parameters

    for p in params['Parameters']:
        if 'prior' in params['Parameters'][p]:
            if 'type' in params['Parameters'][p]['prior']:
                pass
            else:
                raise KeyError('Please provide a type of prior (i.e. uniform, normal).')
        elif 'fixed' in params['Parameters'][p]:
            pass
        else:
            raise KeyError('Please provide valid parameter information (i.e. prior or fixed).')

    # Sampler

    if 'name' not in params['Sampler']:
        params['Sampler']['name'] = 'zeus'
    else:
        if params['Sampler']['name'] not in ['zeus', 'emcee', 'light', 'demc', 'dynesty']:
            raise KeyError('Please use a valid sampler (i.e. zeus, light, emcee, demc).')

    if 'ndim' not in params['Sampler']:
        
        ndim = 0
        for p in params['Parameters']:
            ndim += 1
        params['Sampler']['ndim'] = ndim

    if 'nwalkers' not in params['Sampler']:
        nwalkers = int(2.5 * ndim)
        if nwalkers % 2 == 1:
            nwalkers += 1
        params['Sampler']['nwalkers'] = nwalkers

    if 'nchains' not in params['Sampler']:
        params['Sampler']['nchains'] = 2

    if 'ncheck' not in params['Sampler']:
        params['Sampler']['ncheck'] = 100

    if 'miniter' not in params['Sampler']:
        params['Sampler']['miniter'] = 0

    if 'maxiter' not in params['Sampler']:
        params['Sampler']['maxiter'] = np.inf

    if 'maxcall' not in params['Sampler']:
        params['Sampler']['maxcall'] = np.inf

    if 'thin' not in params['Sampler']:
        params['Sampler']['thin'] = 1

    if 'initial' not in params['Sampler']:
        params['Sampler']['initial'] = 'ellipse'
    else:
        if params['Sampler']['initial'] not in ['ellipse', 'laplace', 'prior']:
            raise KeyError('Please use a valid initialization strategy for the walkers (i.e. ellipse, laplace, prior).')

    # Diagnostics
    epsilon = 0.03
    nact = 20
    dact = 0.01

    if 'Gelman-Rubin' not in params['Diagnostics']:
        params['Diagnostics']['Gelman-Rubin'] = {'use' : True, 'epsilon' : epsilon}

    if 'Autocorrelation' not in params['Diagnostics']:
        params['Diagnostics']['Autocorrelation'] = {'use': True, 'nact' : nact, 'dact' : dact}

    if 'use' not in params['Diagnostics']['Gelman-Rubin']:
        params['Diagnostics']['Gelman-Rubin']['use'] = True

    if 'epsilon' not in params['Diagnostics']['Gelman-Rubin']:
        params['Diagnostics']['Gelman-Rubin']['epsilon'] = epsilon

    if 'use' not in params['Diagnostics']['Autocorrelation']:
        params['Diagnostics']['Autocorrelation']['use'] = True

    if 'nact' not in params['Diagnostics']['Autocorrelation']:
        params['Diagnostics']['Autocorrelation']['nact'] = nact

    if 'dact' not in params['Diagnostics']['Autocorrelation']:
        params['Diagnostics']['Autocorrelation']['dact'] = dact

    # Output

    if 'directory' not in params['Output']:
        params['Output']['directory'] = './chains/'

    if params['Output']['directory'][-1] != '/':
        params['Output']['directory'] += '/'

    if 'tag' not in params['Output']:
        params['Output']['tag'] = 'run'
    
    return params