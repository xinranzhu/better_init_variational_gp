import torch
import pickle as pkl
import argparse
from scipy.io import loadmat
from math import floor
import numpy as np
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
     
    seed = args["seed"]
    path = f'../results/{obj_name}-{dim}_{method_name}_m{npoints}_{expid}_{seed}.pkl'
    pkl.dump(res, open(path, 'wb'))
    print("Results saved to ", path)

def load_data(data_dir='../uci/', dataset="3droad", seed=0, device="cuda"):
    assert device == "cuda" 
    torch.manual_seed(seed) 

    data = torch.Tensor(loadmat(data_dir + dataset + '.mat')['data'])
    X = data[:, :-1]

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        print("Removed %d dimensions with no variance" % (X.size(1) - int(good_dimensions.sum())))
        X = X[:, good_dimensions]

    # if dataset in ['keggundirected', 'slice']:
    #     X = torch.Tensor(SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy()))

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y = data[:, -1]
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.75 * X.size(0)))
    valid_n = int(floor(0.10 * X.size(0)))

    train_x = X[:train_n, :].contiguous().cuda()
    train_y = y[:train_n].contiguous().cuda()

    valid_x = X[train_n:train_n+valid_n, :].contiguous().cuda()
    valid_y = y[train_n:train_n+valid_n].contiguous().cuda()

    test_x = X[train_n+valid_n:, :].contiguous().cuda()
    test_y = y[train_n+valid_n:].contiguous().cuda()
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
