import torch
import sys
import math
import argparse
import numpy as np
import time 

sys.path.append("../src")
sys.path.append("../src/opt")
from splines import spline_K, Dspline_K, spline_fit, rms_vs_truth, spline_eval
from spline_rproj import spline_Jproj, spline_rproj
from fwd_selection import spline_forward_regression
from levenberg_marquardt import levenberg_marquardt
from kernels import SEKernel # TODO: add matern
from utils import store

torch.set_default_dtype(torch.float64) 
torch.set_printoptions(precision=8)
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: add more args
parser = argparse.ArgumentParser(description="parse args")
parser.add_argument("--obj_name", type=str, default="3droad")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--expid", type=str, default="TEST")
parser.add_argument("--kernel_type", type=str, default="SE")
parser.add_argument("--num_inducing", type=int, default=50)

args =  vars(parser.parse_args())
obj_name = args["obj_name"]
dim = args["dim"]
expid = args["expid"]
kernel_type = args["kernel_type"]
num_inducing = args["num_inducing"]

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
train_x = np.loadtxt(f'../data/{obj_name}-{dim}_xx_data.csv', delimiter=",",dtype='float')
test_x = np.loadtxt(f'../data/{obj_name}-{dim}_xx_truth.csv', delimiter=",",dtype='float')
train_y = np.loadtxt(f'../data/{obj_name}-{dim}_y_data.csv', delimiter=",",dtype='float')
test_y = np.loadtxt(f'../data/{obj_name}-{dim}_y_truth.csv', delimiter=",",dtype='float')
train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)
train_n = train_x.shape[0]
test_n = test_x.shape[0]

if device == "cuda":
    train_x = train_x.cuda()
    train_y = train_y.cuda()

print(f"Dataset: {obj_name}, train_n: {train_n}, test_n:{test_n}, num_inducing: {num_inducing}.")

theta_init = 2.0 
sigma = 0.1
rtol = 1e-4
tau = 0.05
theta_penalty = 10.0
lm_nsteps = 150
fwd_num_start = 3
ncand = 10000
count_random_max = 10
max_obj_tol = 1e-6
args["theta_init"] = theta_init
args["sigma"] = sigma
args["rtol"] = rtol
args["tau"] = tau
args["theta_penalty"] = theta_penalty
args["lm_nsteps"] = lm_nsteps
args["fwd_num_start"] = fwd_num_start
args["ncand"] = ncand
args["count_random_max"] = count_random_max
verbose=False

# Forward regression 
method = "fwd"
start = time.time()
ufwd = spline_forward_regression(train_x, train_y, fwd_num_start, num_inducing, 
    theta_init, phi, ncand=ncand, verbose=verbose, 
    count_random_max=count_random_max, max_obj_tol=max_obj_tol)
cfwd = spline_fit(ufwd, train_x, train_y, theta_init, phi, sigma=sigma)
end = time.time()
time_fwd = end-start
args["time"] = time_fwd
r = rms_vs_truth(ufwd, cfwd, theta_init, phi, test_x, test_y)
print(f"Using {method}, RMS: {r:.4f}, norm(c): {torch.norm(cfwd):.2f}, time cost: {time_fwd:.2f} sec.")
store(obj_name, method, num_inducing, cfwd, ufwd, theta_init, phi, sigma, expid=expid, args=args)

# LM
def fproj_test(u, theta):
    if u.dim() == 1:
        u = u.reshape(1,-1)
    u = u.to(train_x.device)
    return spline_rproj(u, train_x, train_y, theta, phi, sigma=sigma)

def Jproj_test(u, theta):
    if u.dim() == 1:
        u = u.reshape(1,-1)
    u = u.to(train_x.device)
    return spline_Jproj(u, train_x, train_y, theta,
       phi, Drho_phi, Dtheta_phi, sigma=sigma)

if theta_penalty > 0:
    def fproj_test(u, theta):
        if u.dim() == 1:
            u = u.reshape(1,-1)
        u = u.to(train_x.device)
        r = spline_rproj(u, train_x, train_y, theta, phi, sigma=sigma)
        tail = torch.tensor(theta_penalty*theta).reshape(-1).to(train_x.device)
        res = torch.cat([r, tail])
        return res
    
    def Jproj_test(u, theta):
        if u.dim() == 1:
            u = u.reshape(1,-1)
        u = u.to(train_x.device)
        J = spline_Jproj(u, train_x, train_y, theta, phi, Drho_phi, Dtheta_phi, sigma=sigma)
        t = torch.zeros(J.shape[1]).to(train_x.device)
        t[-1] = theta_penalty
        return torch.cat([J, t.reshape(1,-1)], dim=0)

method = "lm"
start = time.time()
ulm, thetalm, norm_hist = levenberg_marquardt(fproj_test, Jproj_test, ufwd, theta_init,
    rtol=rtol, tau=tau, nsteps=lm_nsteps)
clm = spline_fit(ulm, train_x, train_y, thetalm, phi, sigma=sigma)
end = time.time()
time_lm = end-start
args["time"] = time_lm
r = rms_vs_truth(ulm, clm, thetalm, phi, test_x, test_y)
print(f"Uisng {method}, RMS: {r:.4f}, norm(c): {torch.norm(clm):.2f}, time cost: {time_lm:.2f} sec.")
store(obj_name, method, num_inducing, clm, ulm, thetalm, phi, sigma, expid=expid, args=args)






