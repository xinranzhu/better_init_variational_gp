import torch
import gpytorch
import sys
import math
import argparse
import numpy as np
import time 
import pickle as pkl
from sklearn.cluster import KMeans
sys.path.append("../src")
sys.path.append("../src/opt")
from splines2 import spline_fit, rms_vs_truth
from levenberg_marquardt import levenberg_marquardt
from utils import store2, load_data, load_data_old


# TODO: add more args
parser = argparse.ArgumentParser(description="parse args")
parser.add_argument("--obj_name", type=str, default="3droad")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--expid", type=str, default="TEST")
parser.add_argument("--kernel_type", type=str, default="SE")
parser.add_argument("--num_inducing", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--init", type=str, default="kmeans")
parser.add_argument("--noise", type=float, default=0.1)


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



# loda data
train_x, train_y, val_x, val_y, test_x, test_y = load_data_old(obj_name, dim, seed=seed)
if device == "cuda":
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    
print(f"Dataset: {obj_name}, train_n: {train_x.shape[0]}  test_n:{test_x.shape[0]}  num_inducing: {num_inducing}.")

sigma = math.sqrt(args["noise"])
rtol = 1e-4
tau = 0.05
theta_penalty = 10.0
lm_nsteps = 150
fwd_num_start = 3
ncand = 8000
count_random_max = 10
max_obj_tol = 1e-6
args["sigma"] = sigma
args["rtol"] = rtol
args["tau"] = tau
args["lm_nsteps"] = lm_nsteps
args["fwd_num_start"] = fwd_num_start
args["ncand"] = ncand
args["count_random_max"] = count_random_max
verbose=False

print("args: ", args)
sys.stdout.flush()

kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())



start=time.time()
assert init == "kmeans"
method="kmeans"
try: 
    res = pkl.load(open(f'../results/{obj_name}-{dim}_kmeans_m{num_inducing}_{expid}_{seed}.pkl', 'rb'))
    u_init = res["u"].to(device=train_x.device)
    print("Loaded kmeans results")
    sys.stdout.flush()
except:
    xk = train_x.cpu().numpy()
    kmeans = KMeans(n_clusters=num_inducing, random_state=seed).fit(xk)
    u_init = torch.tensor(kmeans.cluster_centers_).to(device=train_x.device)

c_init = spline_fit(u_init, train_x, train_y, kernel, sigma=sigma)
end = time.time()
time_init = end-start
args["time"] = time_init
sys.stdout.flush()
print(f"Using {method},  norm(c): {torch.norm(c_init):.2f}  time cost: {time_init:.2f} sec.")
store2(obj_name, method, num_inducing, c_init.cpu(), u_init.cpu(), kernel, sigma, 
    expid=expid, args=args, 
    train_x=train_x.cpu(), test_x=test_x, test_y=test_y)



method = f"lm-{init}"
start = time.time()
output_steps=np.arange(10, lm_nsteps, step=20, dtype=int)
(ulm, lengthscale), norm_hist, _ = levenberg_marquardt(
    u_init, train_x, train_y, kernel, rtol=rtol, tau=tau, sigma=sigma,
    nsteps=lm_nsteps, output_steps=output_steps, log_each_step=True)
kernel.base_kernel.raw_lengthscale = lengthscale

clm = spline_fit(ulm, train_x, train_y, kernel, sigma=sigma)
end = time.time()
time_lm = end-start
args["time"] = time_lm
r = rms_vs_truth(ulm, clm, kernel, test_x, test_y)
args["r"] = r
print(f"Using {method}, RMS: {r:.4e}  norm(c): {torch.norm(clm):.2f}  time cost: {time_lm:.2f} sec.")

sys.stdout.flush()
store2(obj_name, method, num_inducing, clm.cpu(), ulm.cpu(), kernel, sigma, expid=expid, args=args, 
    train_x=train_x.cpu(), test_x=test_x, test_y=test_y)






