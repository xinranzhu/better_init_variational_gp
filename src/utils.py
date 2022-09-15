import torch
import pickle as pkl
import argparse
from splines import spline_K

def store(obj_name, method_name, npoints, c, u, theta, phi, sigma, expid=None):
    dim = u.shape[1]
    Kuu = spline_K(u, u, theta, phi)
    jitter = torch.diag(1e-8*torch.ones(Kuu.shape[0])).to(device=Kuu.device)
    L = torch.linalg.cholesky(Kuu + jitter)
    c = torch.matmul(L.T, c)
    res = {"u": u.cpu(), "c": c.cpu(), "theta": theta, "sigma": sigma}
    pkl.dump(res, open(f'../data/{obj_name}-{dim}_{method_name}_m{npoints}_{expid}.pkl', 'wb'))



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
