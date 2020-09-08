import numpy as np

def get_default(params):

    # General

    if 'Likelihood' not in params:
        raise KeyError('Please provide information about the Likelihood.')

    if 'Parameters' not in params:
        raise KeyError('Please provide information about the Parameters.')

    if 'Sampler' not in params:
        raise KeyError('Please provide information about the Sampler configuration.')

    if 'Diagnostics' not in params:
        params['Diagnostics'] = {'Gelman-Rubin': {'use' : True, 'epsilon' : 0.05},
                                 'Autocorrelation': {'use': True, 'nact' : 10, 'dact' : 0.01}}

    if 'Output' not in params:
        params['Output'] = './'

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
        if params['Sampler']['name'] not in ['zeus', 'emcee', 'dynesty']:
            raise KeyError('Please use a valid sampler (i.e. zeus, emcee).')

    if 'ndim' not in params['Sampler']:
        raise KeyError('Please provide the number of dimensions ndim.')

    if 'nwalkers' not in params['Sampler']:
        raise KeyError('Please provide the number of walkers nwalkers.')

    if 'nchains' not in params['Sampler']:
        raise KeyError('Please provide the number of chains nchains.')

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

    if 'Gelman-Rubin' not in params['Diagnostics']:
        params['Diagnostics']['Gelman-Rubin'] = {'use' : True, 'epsilon' : 0.05}

    if 'Autocorrelation' not in params['Diagnostics']:
        params['Diagnostics']['Autocorrelation'] = {'use': True, 'nact' : 10, 'dact' : 0.03}

    if 'use' not in params['Diagnostics']['Gelman-Rubin']:
        params['Diagnostics']['Gelman-Rubin']['use'] = True

    if 'epsilon' not in params['Diagnostics']['Gelman-Rubin']:
        params['Diagnostics']['Gelman-Rubin']['epsilon'] = 0.05

    if 'use' not in params['Diagnostics']['Autocorrelation']:
        params['Diagnostics']['Autocorrelation']['use'] = True

    if 'nact' not in params['Diagnostics']['Autocorrelation']:
        params['Diagnostics']['Autocorrelation']['nact'] = 10

    if 'dact' not in params['Diagnostics']['Autocorrelation']:
        params['Diagnostics']['Autocorrelation']['dact'] = 0.03

    # Output

    if params['Output'][-1] != '/':
        params['Output'] += '/'

    return params