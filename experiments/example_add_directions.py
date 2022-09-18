import torch
import sys
import math
import argparse
import numpy as np
import pickle as pkl
from sklearn.cluster import DBSCAN
from scipy import spatial

sys.path.append("../src")
sys.path.append("../src/opt")
from splines import spline_K,spline_Kuu, spline_fit, rms_vs_truth, spline_eval
from kernels import SEKernel # TODO: add matern
from utils import store
from process_directions import generate_directions, re_index

torch.set_default_dtype(torch.float64) 
torch.set_printoptions(precision=8)
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: add more args
parser = argparse.ArgumentParser(description="parse_args")
parser.add_argument("--obj_name", type=str, default="3droad")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--expid", type=str, default="TEST")
parser.add_argument("--kernel_type", type=str, default="SE")
parser.add_argument("--num_inducing", type=int, default=50)
parser.add_argument("--method", type=str, default="lm")

args =  vars(parser.parse_args())
obj_name = args["obj_name"]
dim = args["dim"]
expid = args["expid"]
kernel_type = args["kernel_type"]
num_inducing = args["num_inducing"]
method = args["method"]

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
    test_x = test_x.cuda()
    test_y = test_y.cuda()

print(f"Dataset: {obj_name}, train_n: {train_n}, test_n:{test_n}, num_inducing: {num_inducing}.")

# read results
res = pkl.load(open(f'../data/{obj_name}-{dim}_{method}_m{num_inducing}_{expid}.pkl', 'rb'))
u = res["u"].to(device=train_x.device)
c = res["c"].to(device=train_x.device)
theta = res["theta"]
sigma = res["sigma"]



# check previous fitting results without directions
c = spline_fit(u, train_x, train_y, theta, phi, sigma=sigma)
r = rms_vs_truth(u, c, theta, phi, test_x, test_y)
print(f"Using {method}, RMS: {r}, norm(c): {torch.norm(c)}")


# cluster detection 
V, idx_to_remove, num_directions = generate_directions(u.cpu(), q=0.01)
u2, V2 = re_index(u, V, idx_to_remove)

# check new fitting results with directions
u2 = torch.tensor(u2).to(device=train_x.device)
c2 = spline_fit(u2, train_x, train_y, theta, phi, sigma=sigma, V=V2)
r = rms_vs_truth(u2, c2, theta, phi, test_x, test_y, V=V2)
print(f"Adding {num_directions} directions, RMS: {r}, norm(c): {torch.norm(c2)}")



    


