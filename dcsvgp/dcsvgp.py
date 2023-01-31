
import sys
import time
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategyDecoupledConditionals
from torch.utils.data import TensorDataset, DataLoader

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, kernel_type='se', 
        learn_inducing_locations=True, 
        ):
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        covar_module_mean = gpytorch.kernels.RBFKernel()
        variational_strategy = VariationalStrategyDecoupledConditionals(self, inducing_points, 
                                                   variational_distribution, covar_module_mean,
                                                   learn_inducing_locations=learn_inducing_locations)
        super(GPModel, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if kernel_type == 'se':
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_type == 'matern1/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        elif kernel_type == 'matern3/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == 'matern5/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
         
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(model, train_x, train_y, 
    num_epochs=1000, train_batch_size=1024,
    lr=0.01, gamma=1.0,
    elbo_beta=1.0,
    mll_type="ELBO",
    device="cpu", tracker=None, 
    save_model=True, save_path=None,
    test_x=None, test_y=None,
    val_x=None, val_y=None, 
    load_run_path=None,
    save_u=False,obj_name=None,
    lengthscale_only=False,
    verbose=False,
    alpha=-1, idx=0,
    ):

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    previous_epoch = 0
    if load_run_path is not None:
        print("Loading model ", load_run_path)
        last_run = torch.load(load_run_path)
        model.load_state_dict(last_run["model"])
        previous_epoch = last_run["epoch"]
        if lengthscale_only:
            hypers = {}
            hypers['covar_module.raw_lengthscale'] =  torch.tensor(0.)
            hypers['variational_strategy.covar_module_mean.raw_lengthscale'] =  torch.tensor(0.)
            model.initialize(**hypers)
            previous_epoch = 0

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))


    optimizer = torch.optim.Adam([ 
        {'params': model.parameters()},   
    ], lr=lr)

    if lengthscale_only:
        optimizer = torch.optim.Adam([
                {'params': model.variational_strategy.covar_module_mean.raw_lengthscale}, 
                {'params': model.covar_module.raw_lengthscale}, 
            ], lr=lr) 

    milestones = [int(num_epochs*len(train_loader)/3), int(2*num_epochs*len(train_loader)/3)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
       

    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta, alpha=alpha)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta, alpha=alpha)

    min_val_rmse = float("Inf")
    min_val_nll = float("Inf")
    model.train()


    for i in range(num_epochs-previous_epoch):
        for k, (x_batch, y_batch) in enumerate(train_loader):
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            optimizer.zero_grad()
            res, var_k, var_q = model(x_batch)
            output = model.likelihood(res)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        means = output.mean.cpu()
        stds  = output.variance.sqrt().cpu()
        rmse = torch.mean((means - y_batch.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch.cpu()).mean()
        S = model.variational_strategy._variational_distribution.chol_variational_covar
        if tracker is not None:
            tracker.log({
                "loss": loss.item(), 
                "training_rmse": rmse,
                "training_nll": nll,   
                "noise": model.likelihood.noise.item(),
                "train_var": output.variance.cpu()[idx],
                "var_k": var_k[idx],
                "var_q": var_q[idx],
                "pred_mean": output.mean.cpu()[idx],
                "true_y": y_batch[idx],
                "ls_mean":  model.variational_strategy.covar_module_mean.lengthscale.item(),
                "ls": model.covar_module.lengthscale.item(),
                "train_var_denoise": res.variance.cpu()[idx],
                "train_var_comp": var_k[idx] + var_q[idx], 
                "S_mean": S.mean(),
            }, step=i+previous_epoch)
        if i % 30 == 0:
            print(f"loss: {loss.item():.3f}, noise: {model.likelihood.noise.item():.2e}, ls_mean: {model.variational_strategy.covar_module_mean.lengthscale.item():.3f}, ls_covar: {model.covar_module.lengthscale.item():.3f}")
            print(f"train_rmse: {rmse:.2e}, train_nll: {nll:.2e}.")
            print(f"train_var: {output.variance.cpu()}")
            print(f"var_k: {var_k}")
            print(f"var_q: {var_q}")
            print(f"S: mean {S.mean():.2e} median {S.median():.2e} min {S.min():.2e} max {S.max():.2e}.")
            if val_x is not None:
                min_val_rmse, min_val_nll = val_gp(model, val_x, val_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                    min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                    )
            if test_x is not None:
                _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
                model.train()
            if save_model:
                state = {"model": model.state_dict(), "epoch": i}
                torch.save(state, f'{save_path}.model')

    if val_x is not None:
        min_val_rmse, min_val_nll = val_gp(model, val_x, val_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i, 
                    min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                    )
    if test_x is not None:
        _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,
            test_batch_size=1024, device=device, tracker=tracker, step=i)
        print(f"\nLast testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")

    print("model.parameters: ", list(model.named_parameters()))
    
    print("model.mean_lengthscale: ", model.variational_strategy.covar_module_mean.lengthscale)
    print("model.lengthscale: ", model.covar_module.lengthscale)

    # save inducing locations
    if save_u:
        import pickle as pkl
        pkl.dump(model.variational_strategy.inducing_points.detach().cpu(), open(f'./u_dcsvgp_{obj_name}.pkl', 'wb'))
       
    if save_model:
        state = {"model": model.state_dict(), "epoch": i}
        torch.save(state, f'{save_path}.model')

    return model


def eval_gp(model, test_x, test_y,
    test_batch_size=1024, device="cpu", tracker=None, step=0):

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    model.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    var_ks = torch.tensor([0.])
    var_qs = torch.tensor([0.])
    start = time.time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            res, var_k, var_q = model(x_batch)
            preds = model.likelihood(res)
            if device == "cuda":
                means = torch.cat([means, preds.mean.cpu()])
                variances = torch.cat([variances, preds.variance.cpu()])
            else:
                means = torch.cat([means, preds.mean])
                variances = torch.cat([variances, preds.variance])
            var_ks = torch.cat([var_ks, var_k])
            var_qs = torch.cat([var_qs, var_q])
    end = time.time()
    testing_time = end - start

    means = means[1:]
    variances = variances[1:]
    
    rmse = torch.mean((means - test_y.cpu())**2).sqrt()
    nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y.cpu()).mean()
    if tracker is not None:
        tracker.log({
            "testing_rmse":rmse, 
            "testing_nll":nll,
        }, step=step)
    return means, variances, rmse, nll, testing_time




def val_gp(model, val_x, val_y,
    test_batch_size=1024, device="cpu", tracker=None, step=0,
    min_val_rmse=None, min_val_nll=None):

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    model.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])

    with torch.no_grad():
        for x_batch, _ in val_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
            res, var_k, var_q = model(x_batch)
            preds = model.likelihood(res)
            if device == "cuda":
                means = torch.cat([means, preds.mean.cpu()])
                variances = torch.cat([variances, preds.variance.cpu()])
            else:
                means = torch.cat([means, preds.mean])
                variances = torch.cat([variances, preds.variance])

    means = means[1:]
    variances = variances[1:]
    
    rmse = torch.mean((means - val_y.cpu())**2).sqrt()
    nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(val_y.cpu()).mean()
    min_val_nll = min(min_val_nll, nll)
    min_val_rmse = min(min_val_rmse, rmse)
    if tracker is not None:
        tracker.log({
            "val_rmse": min_val_rmse, 
            "val_nll": min_val_nll,
        }, step=step)
    return min_val_rmse, min_val_nll 
