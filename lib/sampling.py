import numpy as np
from numpy.random import RandomState

def sample(seed, model_params, percentage_tosample):
    """
    returns a stratified sample of the model params
    """
    result = []
    rng = RandomState(seed)
    for k, v in model_params.items():
        result.append(rng.choice(np.ravel(v),
                                 size=int(len(np.ravel(v))*percentage_tosample),
                                replace=False))

    result = np.hstack(result)

    return result