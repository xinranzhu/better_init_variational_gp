import sys
import time
import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, inducing_points, kernel_type='se', ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood() 

        super(GPModel, self).__init__(train_x, train_y, likelihood)
        
        # self.base_covar_module = ScaleKernel(RBFKernel())
        if kernel_type == 'se':
            self.base_covar_module = ScaleKernel(RBFKernel())
        elif kernel_type == 'matern1/2':
            self.base_covar_module = MaternKernel(nu=0.5)
        elif kernel_type == 'matern3/2':
            self.base_covar_module = MaternKernel(nu=1.5)
        elif kernel_type == 'matern5/2':
            self.base_covar_module = MaternKernel(nu=2.5)
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)
        self.mean_module = ConstantMean()
        self.likelihood = likelihood
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
def train_gp(model, train_x, train_y, 
    num_epochs=1000,
    lr=0.01,
    scheduler="multistep", gamma=1.0,
    mll_type="ELBO",
    device="cpu", tracker=None, 
    save_model=True, save_path=None,
    test_x=None, test_y=None,
    val_x=None, val_y=None,
    load_run_path=None,
    debug=False, verbose=True
    ):


    if device == "cuda":
        try:
            model = model.to(device=torch.device("cuda"))
            train_x = train_x.cuda()
            train_y = train_y.cuda()
        except:
            print("Training data too large to fit in CUDA memory. Using CPU instead.")

    previous_epoch = 0
    if load_run_path is not None:
        last_run = torch.load(load_run_path)
        model.load_state_dict(last_run["model"])
        previous_epoch = last_run["epoch"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    assert scheduler == "multistep" and mll_type == "ELBO"
    milestones = [int(num_epochs/3), int(2*num_epochs/3)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)


    start = time.time()
    min_val_rmse = float("Inf")
    min_val_nll = float("Inf")
    model.train()
    for i in range(num_epochs-previous_epoch):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        means = output.mean.cpu()
        stds  = output.variance.sqrt().cpu()
        rmse = torch.mean((means - train_y.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(train_y.cpu()).mean()

        if tracker is not None:
            tracker.log({
                "loss": loss.item(), 
                "training_rmse": rmse,
                "training_nll": nll,    
            }, step=i+previous_epoch)

        _, _, val_rmse, val_nll = eval_gp(model, val_x, val_y,
        device=device, tracker=tracker, step=i+previous_epoch, name="val")
        min_val_rmse = min(min_val_rmse, val_rmse)
        min_val_nll = min(min_val_nll, val_nll)

        _, _, test_rmse, test_nll, = eval_gp(model, test_x, test_y,
            device=device, tracker=tracker, step=i+previous_epoch, name="testing")
        
        model.train()

        if save_model:
            state = {"model": model.state_dict(), "epoch": i}
            torch.save(state, f'{save_path}.model')
        if debug:
            print("u.grad, ", model.variational_strategy.inducing_points.grad.abs().mean().item())
            print("covar.grad, ", model.variational_strategy._variational_distribution.chol_variational_covar.grad.abs().mean().item())
            print("mean.grad, ", model.variational_strategy._variational_distribution.variational_mean.grad.abs().mean().item())
            print("raw_lengthscale.grad, ", model.covar_module.raw_lengthscale.grad.abs().mean().item())
            print("mean_const.grad, ", model.mean_module.constant.grad.abs().mean().item())
            print("raw_noise.grad, ", model.likelihood.raw_noise.grad.abs().mean().item())
            sys.stdout.flush()
        if verbose:
            print(f"\n\nEpoch {i}, loss: {loss.item():.3f}, nll: {nll:.3f}, rmse: {rmse:.3e}")
            print(f"testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")
            sys.stdout.flush()

    end = time.time()
    training_time = end - start
    _, _, val_rmse, val_nll = eval_gp(model, val_x, val_y,
        device=device, tracker=tracker, step=i+previous_epoch, name="val")
    min_val_rmse = min(min_val_rmse, val_rmse)
    min_val_nll = min(min_val_nll, val_nll)
    _, _, test_rmse, test_nll, = eval_gp(model, test_x, test_y,
        device=device, tracker=tracker, step=i+previous_epoch, name="testing")
    print(f"\nLast testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")
    if save_model:
        state = {"model": model.state_dict(), "epoch": i}
        torch.save(state, f'{save_path}.model')

    if tracker is not None:
        tracker.log({
            "training_time":training_time,       
        })
    return model, training_time

def eval_gp(model, test_x, test_y,
    device="cpu", tracker=None, step=0, name="testing"):

    if device == "cuda":
        try:
            model = model.to(device=torch.device("cuda"))
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        except:
            print("Testing data too large to fit in CUDA memory. Using CPU instead.")
        
    model.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
        
    with torch.no_grad():
        preds = model(test_x)
        means = torch.cat([means, preds.mean.cpu()])
        variances = torch.cat([variances, preds.variance.cpu()])

    means = means[1:]
    variances = variances[1:]
    
    rmse = torch.mean((means - test_y.cpu())**2).sqrt()
    nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y.cpu()).mean()
    if tracker is not None:
        tracker.log({
            f"{name}_rmse":rmse, 
            f"{name}_nll":nll,
        }, step=step)

    return means, variances, rmse, nll
