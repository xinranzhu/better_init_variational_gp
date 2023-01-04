from synthetic_functions import *
from rescale import *
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_synthetic_data(test_fun, n, **kwargs):
    """
    load synthetic data 
    Input: 
        test_fun: a modified Botorch test function
        n: number of datapoints
    Output: 
        x: torch tensor, random data from unit cube
        y: torch tensor, normalized and rescaled labels (w/ or w/o derivatives)
    """
    torch.random.manual_seed(kwargs["seed"])
    dim = test_fun.dim
    x_unit = torch.rand(n,dim)
    # evaluate in the true range
    lb, ub = test_fun.get_bounds()
    x = from_unit_cube(x_unit, lb, ub)
    if kwargs["derivative"]:
        y = test_fun.evaluate_true_with_deriv(x)
    else:
        y = test_fun.evaluate_true(x)
    # normalize y values (with or without derivatives)
    normalize(y, **kwargs)
    if kwargs["derivative"]:
        # mapping derivative values to unit cube
        f = y[..., 0].reshape(len(y),1)
        g = y[..., 1:].reshape(len(y),-1)
        g *= (ub - lb)
        y = torch.cat([f, g], 1)

    # add scaling factors to info_dict for further accurate plot
    info_dict = {}
    return x_unit, y, info_dict