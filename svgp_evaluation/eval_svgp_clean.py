import sys
import torch
import pickle as pkl
sys.path.append("./models")
from svgp import GPModel, eval_gp
sys.path.append("../src")
from utils import load_data_1d
import ipdb

torch.set_default_dtype(torch.float64) 
device = torch.device("cpu")

obj_name = "1Dtoy"
dim = 1
kernel_type = "SE" # use squared exponential kernel 
num_inducing = 20
seed = 214 # random seed

train_x, train_y, test_x, test_y = load_data_1d(seed=seed)
print(f"Dataset: {obj_name}, train_n: {train_x.shape[0]}  test_n:{test_x.shape[0]}  num_inducing: {num_inducing}.")
print(train_y[:3])
print(test_y[:3])

res = pkl.load(open("../experiments/initialization_results.pkl", "rb"))
u0 = res["u"].to(device=device)
c = res["c"]
sigma = res["sigma"]
theta = res["theta"]
Sbar = res["Sbar"]
Lbar = torch.linalg.cholesky(Sbar)
assert u0.shape[0] == num_inducing and u0.shape[1] == dim


u0 = torch.tensor(u0)
model = GPModel(inducing_points=u0)
# load initialization 
hypers = {}
hypers['covar_module.lengthscale'] =  torch.tensor(theta)
hypers["likelihood.noise_covar.noise"] = torch.tensor(sigma**2)
hypers["variational_strategy._variational_distribution.chol_variational_covar"] = Lbar.to(u0.device)
hypers["variational_strategy._variational_distribution.variational_mean"] = c.to(u0.device)
model.variational_strategy.variational_params_initialized = torch.tensor(1)
model.initialize(**hypers)
# evaluate before training for sanity check 
means, variances, rmse, test_nll, _ = eval_gp(model, test_x, test_y, device=device)
print(f"initial test rmse: {rmse:.4e}, test nll: {test_nll:.4e}")
        
