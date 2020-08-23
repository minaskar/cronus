import importlib.util
import sys
import inspect


def import_loglikelihood(params):
    function = params['Likelihood']['function']
    path = params['Likelihood']['path']

    spec = importlib.util.spec_from_file_location(function, path)
    logprob = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = logprob
    spec.loader.exec_module(logprob)

    list_of_functions = inspect.getmembers(logprob, inspect.isfunction)

    for i in range(len(list_of_functions)):
        if list_of_functions[i][0] == function:
            return list_of_functions[i][1]