import torch
import sys
import math
import argparse
import numpy as np
import pickle as pkl
from sklearn.cluster import DBSCAN
from scipy import spatial
import time 

sys.path.append("../src")
sys.path.append("../src/opt")
from splines import spline_K,spline_Kuu, spline_fit, rms_vs_truth, spline_eval
from kernels import SEKernel # TODO: add matern
from utils import store, load_data, load_data_old
from process_directions import generate_directions, re_index

parser = argparse.ArgumentParser(description="parse_args")
parser.add_argument("--obj_name", type=str, default="3droad")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--expid", type=str, default="TEST")
parser.add_argument("--kernel_type", type=str, default="SE")
parser.add_argument("--num_inducing", type=int, default=50)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--method", type=str, default="lm-kmeans")
parser.add_argument("--q", type=float, default=0.01)
parser.add_argument("--num_cut", type=int, default=20)


args =  vars(parser.parse_args())
obj_name = args["obj_name"]
dim = args["dim"]
expid = args["expid"]
kernel_type = args["kernel_type"]
num_inducing = args["num_inducing"]
method = args["method"]
seed = args["seed"]

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
data_loader = 0 if obj_name in {"bike", "energy", "protein"} else 1
if data_loader > 0:
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(dataset=obj_name, seed=seed)
else:
    train_x, train_y, val_x, val_y, test_x, test_y = load_data_old(obj_name, dim, seed=seed)

if device == "cuda":
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

print(f"Dataset: {obj_name}, train_n: {train_x.shape[0]}  test_n:{test_x.shape[0]}  num_inducing: {num_inducing}.")

# read results
res = pkl.load(open(f'../results/{obj_name}-{dim}_{method}_m{num_inducing}_{expid}_{seed}.pkl', 'rb'))
u = res["u"].to(device=train_x.device)
c = res["c"].to(device=train_x.device)
theta = res["theta"]
sigma = res["sigma"]
q=args["q"]
num_cut=args["num_cut"]


# check previous fitting results without directions
c = spline_fit(u, train_x, train_y, theta, phi, sigma=sigma)
r = rms_vs_truth(u, c, theta, phi, test_x, test_y)
print(f"Using {method}, RMS: {r:.4e}, stored_r: {res['r']:.4e} norm(c): {torch.norm(c):.2f}.")


# cluster detection 
start = time.time()
V, idx_to_remove, _ = generate_directions(u.cpu(), q=q, num_cut=num_cut)
u2, V2, num_directions = re_index(u, V, idx_to_remove)


# check new fitting results with directions
# u2 = torch.tensor(u2).to(device=train_x.device)
u2 = u2.to(device=train_x.device)
c2 = spline_fit(u2, train_x, train_y, theta, phi, sigma=sigma, V=V2)
end = time.time()
time_cost = end-start
args["time"] = time_cost
r = rms_vs_truth(u2, c2, theta, phi, test_x, test_y, V=V2)
print(f"Adding {num_directions} directions, {u2.shape[0]} inducing points left,  RMS: {r:.4e}  norm(c): {torch.norm(c2):.2f} time_cost: {time_cost:.2f} sec.")
store(obj_name, method, num_inducing, c2, u2, theta, phi, sigma, 
    expid=expid, args=args, V=V2, num_cut=num_cut)




    


