import torch
import sys
import math
import argparse
import numpy as np
import time 
import pickle as pkl
from sklearn.cluster import KMeans
sys.path.append("../src")
sys.path.append("../src/opt")
from splines import spline_K, Dspline_K, spline_fit, rms_vs_truth, spline_eval
from spline_rproj import spline_Jproj, spline_rproj
from fwd_selection import spline_forward_regression
from levenberg_marquardt import levenberg_marquardt
from kernels import SEKernel # TODO: add matern
from utils import store, load_data


# TODO: add more args
parser = argparse.ArgumentParser(description="parse args")
parser.add_argument("--obj_name", type=str, default="3droad")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--expid", type=str, default="TEST")
parser.add_argument("--kernel_type", type=str, default="SE")
parser.add_argument("--num_inducing", type=int, default=50)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--data_loader", type=int, default=0)
parser.add_argument("--init", type=str, default="fwd")

args =  vars(parser.parse_args())
obj_name = args["obj_name"]
dim = args["dim"]
expid = args["expid"]
kernel_type = args["kernel_type"]
num_inducing = args["num_inducing"]
seed = args["seed"]
init = args["init"]

torch.set_default_dtype(torch.float64) 
torch.set_printoptions(precision=8)
device = "cuda" if torch.cuda.is_available() else "cpu"
# get GPU type
devices = [d for d in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]
args["gpu"] = device_names

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
if args["data_loader"] > 0:
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(dataset=obj_name, device=device, seed=seed)
    train_n = train_x.shape[0]
    test_n = test_x.shape[0]
else:
    train_x = np.loadtxt(f'../data/{obj_name}-{dim}_xx_data.csv', delimiter=",",dtype='float')
    test_x = np.loadtxt(f'../data/{obj_name}-{dim}_xx_truth.csv', delimiter=",",dtype='float')
    train_y = np.loadtxt(f'../data/{obj_name}-{dim}_y_data.csv', delimiter=",",dtype='float')
    test_y = np.loadtxt(f'../data/{obj_name}-{dim}_y_truth.csv', delimiter=",",dtype='float')
    train_n = train_x.shape[0]
    test_n = test_x.shape[0]
    X = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    Xy = np.concatenate([X,y.reshape(-1,1)], axis=1)
    np.random.seed(seed)
    np.random.shuffle(Xy)
    train_x = torch.tensor(Xy[:train_n,:-1])
    train_y = torch.tensor(Xy[:train_n,-1])
    test_x = torch.tensor(Xy[train_n:,:-1])
    test_y = torch.tensor(Xy[train_n:,-1])
            
    if device == "cuda":
        train_x = train_x.cuda()
        train_y = train_y.cuda()

print(f"Dataset: {obj_name}, train_n: {train_n}  test_n:{test_n}  num_inducing: {num_inducing}.")

theta_init = 2.0 
sigma = 0.1
rtol = 1e-4
tau = 0.05
theta_penalty = 10.0
lm_nsteps = 150
fwd_num_start = 3
ncand = 8000
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
start=time.time()
if init == "fwd": 
    method = "fwd"
    # try load 
    print("args: ", args)
    try: 
        res = pkl.load(open(f'../results/{obj_name}-{dim}_fwd_m{num_inducing}_{expid}_{seed}.pkl', 'rb'))
        u_init = res["u"].to(device=train_x.device)
        print("Loaded fwd results")
    except:
        start = time.time()
        u_init = spline_forward_regression(train_x, train_y, fwd_num_start, num_inducing, 
            theta_init, phi, ncand=ncand, verbose=verbose, 
            count_random_max=count_random_max, max_obj_tol=max_obj_tol)
else:
    method="kmeans"
    xk = train_x.cpu().numpy()
    kmeans = KMeans(n_clusters=num_inducing, random_state=seed).fit(xk)
    u_init = torch.tensor(kmeans.cluster_centers_).to(device=train_x.device)

c_init = spline_fit(u_init, train_x, train_y, theta_init, phi, sigma=sigma)
end = time.time()
time_init = end-start
args["time"] = time_init
r = rms_vs_truth(u_init, c_init, theta_init, phi, test_x, test_y)
args["r"] = r
print(f"Using {method}, RMS: {r:.4e}  norm(c): {torch.norm(c_init):.2f}  time cost: {time_init:.2f} sec.")
store(obj_name, method, num_inducing, c_init, u_init, theta_init, phi, sigma, expid=expid, args=args)



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
ulm, thetalm, norm_hist = levenberg_marquardt(fproj_test, Jproj_test, u_init, theta_init,
    rtol=rtol, tau=tau, nsteps=lm_nsteps)
clm = spline_fit(ulm, train_x, train_y, thetalm, phi, sigma=sigma)
end = time.time()
time_lm = end-start
args["time"] = time_lm
r = rms_vs_truth(ulm, clm, thetalm, phi, test_x, test_y)
args["r"] = r
print(f"Uisng {method}, RMS: {r:.4e}  norm(c): {torch.norm(clm):.2f}  time cost: {time_lm:.2f} sec.")
store(obj_name, method, num_inducing, clm, ulm, thetalm, phi, sigma, expid=expid, args=args)






