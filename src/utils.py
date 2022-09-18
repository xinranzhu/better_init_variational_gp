import torch
import pickle as pkl
import argparse
from splines import spline_K, spline_Kuu

def store(obj_name, method_name, npoints, c, u, theta, phi, sigma, 
    expid=None, args=None, V=None):

    m, dim = u.shape
    res = {"u": u.cpu(), "theta": theta, "sigma": sigma}
    if V is None:
        Kuu = spline_K(u, u, theta, phi)
    else:
        method_name = method_name + "-directions"
        Kuu = spline_Kuu(u, theta, phi, V=V)
        # process V_mat and inducing_values_num
        V_mat = torch.empty((0, dim))
        inducing_values_num = torch.zeros(m, dtype=int)
        for key in sorted(V.keys()):
            V_mat = torch.cat([V_mat, torch.tensor(V[key])], dim=0)
            inducing_values_num[key] = V[key].shape[0]
        res["V_mat"] = V_mat
        res["inducing_values_num"] = inducing_values_num

    jitter = torch.diag(1e-8*torch.ones(Kuu.shape[0])).to(device=Kuu.device)
    L = torch.linalg.cholesky(Kuu + jitter)
    c = torch.matmul(L.T, c)
    res["c"] = c.cpu()
    if args:
        res.update(args)

    path = f'../data/{obj_name}-{dim}_{method_name}_m{npoints}_{expid}.pkl'
    pkl.dump(res, open(path, 'wb'))
    print("Results saved to ", path)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
