import torch
import sys
import argparse
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
sys.path.append("../src")
sys.path.append("../src/opt")
from splines import spline_K, Dspline_K, spline_fit, rms_vs_truth, spline_eval
from spline_rproj import spline_Jproj, spline_rproj
from levenberg_marquardt import levenberg_marquardt
from kernels import SEKernel 
from utils import store, load_data_1d

torch.set_default_dtype(torch.float64) 

obj_name = "1Dtoy"
dim = 1
kernel_type = "SE" # use squared exponential kernel 
num_inducing = 20
seed = 214 # random seed



# set kernel wrappers 
def phi(rho, theta):
    kernel = eval(f"{kernel_type}Kernel")(theta=theta)
    return kernel.phi(rho)
def Drho_phi(rho, theta):
    kernel = eval(f"{kernel_type}Kernel")(theta=theta)
    return kernel.Drho_phi(rho)
def Dtheta_phi(rho, theta):
    kernel = eval(f"{kernel_type}Kernel")(theta=theta)
    return kernel.Dtheta_phi(rho)


train_x, train_y, test_x, test_y = load_data_1d(seed=seed)
print(f"Dataset: {obj_name}, train_n: {train_x.shape[0]}  test_n:{test_x.shape[0]}  num_inducing: {num_inducing}.")

print(train_y[:3])
print(test_y[:3])

theta_init = 2.0 # initial value for lengthscale
sigma = 0.1 # initial value for noise variance
rtol = 1e-4 # tolerance for levenberg marquardt algorithm (LM)
tau = 0.05  # scaling parameter for levenberg marquardt algorithm (LM)
theta_penalty = 10.0 # scaling factor for theta penalty term in minimization, could be 0
lm_nsteps = 100 # number of steps in levenberg marquardt algorithm (LM)



# Kmeans initialization for LM
xk = train_x.cpu().numpy()
kmeans = KMeans(n_clusters=num_inducing, random_state=seed).fit(xk)
u_init = torch.tensor(kmeans.cluster_centers_).to(device=train_x.device)

c_init = spline_fit(u_init, train_x, train_y, theta_init, phi, sigma=sigma)
test_rmse = rms_vs_truth(u_init, c_init, theta_init, phi, test_x, test_y)
print(f"Using kmeans, test rmse: {test_rmse:.4e}  norm(c): {torch.norm(c_init):.2f}.")



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


ulm, thetalm, norm_hist, res_dict, train_rmse = levenberg_marquardt(fproj_test, Jproj_test, u_init, theta_init,
    rtol=rtol, tau=tau, nsteps=lm_nsteps)
clm = spline_fit(ulm, train_x, train_y, thetalm, phi, sigma=sigma)

test_rmse = rms_vs_truth(ulm, clm, thetalm, phi, test_x, test_y)
# args["r"] = r.cpu()
print(f"Using LM, test_rmse: {test_rmse:.4e}  norm(c): {torch.norm(clm):.2f}, train_rmse:{train_rmse:.2e}.")
store(obj_name, "LM", num_inducing, clm.cpu(), ulm.cpu(), thetalm, phi, sigma, 
    train_x=train_x.cpu(), test_x=test_x, test_y=test_y, CLEAN=True)






