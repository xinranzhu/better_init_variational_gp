import torch
import pickle as pkl
import argparse
from scipy.io import loadmat
from math import floor
import numpy as np
import sys
from splines import spline_K, spline_Kuu, spline_eval

def store(obj_name, method_name, npoints, c, u, theta, phi, sigma, 
    expid=None, args=None, V=None, num_cut=None, train_x=None, test_x=None, test_y=None, k=None):

    m, dim = u.shape
    res = {"u": u.cpu(), "theta": theta, "sigma": sigma}
    if V is None:
        Kuu = spline_K(u, u, theta, phi)
        Kux = spline_K(u, train_x, theta, phi)
        Kxu = Kux.T
    else:
        method_name = method_name + "-directions"
        Kuu = spline_Kuu(u, theta, phi, V=V)
        Kux = spline_K(u, train_x, theta, phi, V=V)
        Kxu = Kux.T
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
    pred_means = spline_eval(test_x, u, c, theta, phi)
    c = torch.matmul(L.T, c) # whitened mean 
    res["c"] = c.cpu()

    # compute the whitened covariance 
    Ktt = spline_K(test_x, test_x, theta, phi)
    jitter2 = torch.diag(1e-4*torch.ones(Ktt.shape[0])).to(device=Ktt.device)
    Ktu = spline_K(test_x, u, theta, phi)
    Kut = Ktu.T
    M = Kuu + Kux @ Kxu/sigma/sigma
    Sopt = Kuu @ (torch.linalg.solve(M, Kuu)) # optimal variational covariance
    Sbar = L.T @ (torch.linalg.solve(M, L)) # whitened Sopt
    interp_term = torch.linalg.solve(L, Kut)
    mid_term = Sbar - torch.eye(m).to(device=Sbar.device)

    pred_covar = Ktt + jitter2 + interp_term.T @ mid_term @ interp_term
    # compute nll for verification
    stds = torch.sqrt(torch.diag(pred_covar)) + sigma
    nll = - torch.distributions.Normal(pred_means, stds).log_prob(test_y).mean()

    res["Sbar"] = Sbar.cpu()
    res["L"] = L.cpu()
    res["nll"] = nll
    if args:
        res.update(args) 
        
    seed = args["seed"]
    if num_cut is None and k is None:
        path = f'../results/{obj_name}-{dim}_{method_name}_m{npoints}_{expid}_{seed}.pkl'
    elif k is None:
        path = f'../results/{obj_name}-{dim}_{method_name}_m{npoints}_{expid}_{seed}_{num_cut}.pkl'
    elif num_cut is None:
        path = f'../results/{obj_name}-{dim}_{method_name}_m{npoints}_{expid}_{seed}_step{k}.pkl'
    else:
        path = f'../results/{obj_name}-{dim}_{method_name}_m{npoints}_{expid}_{seed}_{num_cut}_step{k}.pkl'


    pkl.dump(res, open(path, 'wb'))
    print(f"test_nll: {nll:.3e}, res saved to {path}.")

def load_data(data_dir='../uci/', dataset="3droad", seed=0):
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

    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    valid_x = X[train_n:train_n+valid_n, :].contiguous()
    valid_y = y[train_n:train_n+valid_n].contiguous()

    test_x = X[train_n+valid_n:, :].contiguous()
    test_y = y[train_n+valid_n:].contiguous()

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_data_old(obj_name, dim, seed=0):
    xx_data = np.loadtxt(f'../data/{obj_name}-{dim}_xx_data.csv', delimiter=",",dtype='float')
    xx_truth = np.loadtxt(f'../data/{obj_name}-{dim}_xx_truth.csv', delimiter=",",dtype='float')
    y_data = np.loadtxt(f'../data/{obj_name}-{dim}_y_data.csv', delimiter=",",dtype='float')
    y_truth = np.loadtxt(f'../data/{obj_name}-{dim}_y_truth.csv', delimiter=",",dtype='float')
    train_n = xx_data.shape[0]
    test_n = int(xx_truth.shape[0]*2/3)
    val_n = xx_truth.shape[0] - test_n
    test_n = test_n
    val_n = val_n

    X = np.concatenate([xx_data, xx_truth], axis=0)
    y = np.concatenate([y_data, y_truth], axis=0)
    Xy = np.concatenate([X,y.reshape(-1,1)], axis=1)
    # randomly shuffle the data
    if seed > 0:
        np.random.seed(seed)
        np.random.shuffle(Xy)

    train_x = torch.tensor(Xy[:train_n,:-1])
    train_y = torch.tensor(Xy[:train_n,-1])
    valid_x = torch.tensor(Xy[train_n:,:-1])[:val_n]
    valid_y = torch.tensor(Xy[train_n:,-1])[:val_n]
    test_x = torch.tensor(Xy[train_n:,:-1])[val_n:]
    test_y = torch.tensor(Xy[train_n:,-1])[val_n:]
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


def check_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(f"Total: {(t/1024/1024/1024):.2f} GB, allocated: {(a/1024/1024/1024):.2f} GB, reserved: {(r/1024/1024/1024):.2f} GB" )
    sys.stdout.flush()

