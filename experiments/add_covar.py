import torch
import sys
import math
import argparse
import numpy as np
import time
import pickle as pkl


sys.path.append("../src")
sys.path.append("../src/opt")
from splines import spline_K, Dspline_K, spline_fit, rms_vs_truth, spline_eval
from spline_rproj import spline_Jproj, spline_rproj
from fwd_selection import spline_forward_regression
from levenberg_marquardt import levenberg_marquardt
from kernels import SEKernel # TODO: add matern
from utils import store, load_data, load_data_old, check_cuda_memory

torch.set_default_dtype(torch.float64) 
torch.set_printoptions(precision=8)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="parse_args")
parser.add_argument("--obj_name", type=str, default="3droad")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--expid", type=str, default="TEST")
parser.add_argument("--kernel_type", type=str, default="SE")
parser.add_argument("--num_inducing", type=int, default=50)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--method", type=str, default="lm-kmeans")


args =  vars(parser.parse_args())
obj_name = args["obj_name"]
dim = args["dim"]
expid = args["expid"]
kernel_type = args["kernel_type"]
num_inducing = args["num_inducing"]
method = args["method"]
seed = args["seed"]



# set kernel
def phi(rho, theta):
    kernel = eval(f"{kernel_type}Kernel")(theta=theta)
    return kernel.phi(rho)
def Drho_phi(rho, theta):
    kernel = eval(f"{kernel_type}Kernel")(theta=theta)
    return kernel.Drho_phi(rho)
def Dtheta_phi(rho, theta):
    kernel = eval(f"{kernel_type}Kernel")(theta=theta)
    return kernel.Dtheta_phi(rho)

# loda data
data_loader = 0 if obj_name in {"bike", "energy", "protein"} else 1
if data_loader > 0:
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(dataset=obj_name, seed=seed)
    dim = train_x.shape[1]
else:
    train_x, train_y, val_x, val_y, test_x, test_y = load_data_old(obj_name, dim, seed=seed)
if device == "cuda":
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()
print(f"Dataset: {obj_name}, train_n: {train_x.shape[0]}  test_n:{test_x.shape[0]}  num_inducing: {num_inducing}.")


# read results
path = f'../results/{obj_name}-{dim}_{method}_m{num_inducing}_{expid}_{seed}.pkl'
res = pkl.load(open(path, 'rb'))
print("reading results from :", path)
u = res["u"].to(device=train_x.device)
c = res["c"].to(device=train_x.device)
theta = res["theta"]
sigma = res["sigma"]
m = u.shape[0] # number of inducing points

# check previous fitting results without directions
c = spline_fit(u, train_x, train_y, theta, phi, sigma=sigma)
r = rms_vs_truth(u, c, theta, phi, test_x, test_y)
print(f"Using {method}, RMS: {r:.4e}, stored_r: {res['r']:.4e} norm(c): {torch.norm(c):.2f}.")

# compute covar and store 
Kuu = spline_K(u, u, theta, phi)
jitter = torch.diag(1e-8*torch.ones(Kuu.shape[0])).to(device=Kuu.device)
L = torch.linalg.cholesky(Kuu + jitter)
Kux = spline_K(u, train_x, theta, phi)
Kxu = Kux.T
Ktt = spline_K(test_x, test_x, theta, phi)
jitter2 = torch.diag(1e-4*torch.ones(Ktt.shape[0])).to(device=Ktt.device)
Ktu = spline_K(test_x, u, theta, phi)
Kut = Ktu.T

pred_means = spline_eval(test_x, u, c, theta, phi)
optimal_mean = torch.matmul(L.T, c)

M = Kuu + Kux @ Kxu/sigma/sigma
Sopt = Kuu @ (torch.linalg.solve(M, Kuu)) # optimal variational covariance
Sbar = L.T @ (torch.linalg.solve(M, L)) # whitened Sopt
interp_term = torch.linalg.solve(L, Kut)
mid_term = Sbar - torch.eye(m).to(device=Sbar.device)

pred_covar = Ktt + jitter2 + interp_term.T @ mid_term @ interp_term

# compute nll for verification
stds = torch.sqrt(torch.diag(pred_covar))
nll = - torch.distributions.Normal(pred_means, stds).log_prob(test_y).mean()
print(f"nll: {nll}, res saved to {path}.")

# save Sbar, the whitened covariance 
res["Sbar"] = Sbar.cpu()
res["L"] = L.cpu()
pkl.dump(res, open(path, 'wb'))