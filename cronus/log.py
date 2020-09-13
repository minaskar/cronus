import logging
import sys

def create_logger(module_name, level=logging.INFO):
    """
    Set up the logging channel `module_name`.
    Write to stdout with output level `level`.
    If logging handlers are already registered, no new handlers are
    registered.
    """
    logger = logging.getLogger(str(module_name))
    logger.setLevel(logging.DEBUG)
    first_logger = logger.handlers == []
    
    if first_logger:
        # if it is new, register to write to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('[{}] %(message)s'.format(module_name))
        handler.setFormatter(formatter)
        #logger.setLevel(level)
        logger.addHandler(handler)

    return logger
