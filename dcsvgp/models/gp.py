import sys
import time
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def train_gp(model, likelihood, train_x, train_y, 
    num_epochs=1000, 
    lr=0.01, device="cpu", tracker=None,
    scheduler=None, gamma=0.3,
    test_x=None, test_y=None):

    if device == "cuda":
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.to(device=torch.device("cuda"))
        likelihood = likelihood.to(device=torch.device("cuda"))
        
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        ], lr=lr)

    if scheduler == "multistep":
        milestones = [int(num_epochs/3), int(2*num_epochs/3)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    elif scheduler == None:
        lr_sched = lambda epoch: 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)
    elif scheduler == "lambda":
        lr_sched = lambda epoch: 0.8 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)


    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        means = output.mean.cpu()
        stds  = output.variance.sqrt().cpu()
        rmse = torch.mean((means - train_y.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(train_y.cpu()).mean()
        if tracker is not None:
            tracker.log({
                "loss": loss.item(), 
                "training_rmse": rmse,
                "training_nll": nll, 
                "ls": model.covar_module.lengthscale.mean().item(),
            }, step=i)
        if test_x is not None:
            _, _, test_rmse, test_nll = eval_gp(model, model.likelihood, test_x=test_x, test_y=test_y,
                device=device, tracker=tracker, step=i)
            model.train()
        if i % 100 == 0:
            print(f"Epoch: {i}, loss: {loss.item()}, nll: {nll}, rmse: {rmse}")
            sys.stdout.flush()
        
    
    
    return model, likelihood

def eval_gp(model, likelihood, test_x, test_y, device="cpu",tracker=None, step=0):

    if device == "cuda":
        test_x = test_x.cuda()
        model = model.to(device=torch.device("cuda"))
        likelihood = likelihood.to(device=torch.device("cuda"))

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    with torch.no_grad():
        preds = likelihood(model(test_x))
        if device == "cuda":
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])
        else:
            means = torch.cat([means, preds.mean])
            variances = torch.cat([variances, preds.variance])
    
    means = means[1:]
    variances = variances[1:]
    
    rmse = torch.mean((means - test_y.cpu())**2).sqrt()
    nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y.cpu()).mean()
    if tracker is not None:
        tracker.log({
            "testing_rmse":rmse, 
            "testing_nll":nll,
        }, step=step)
    return means, variances, rmse, nll
