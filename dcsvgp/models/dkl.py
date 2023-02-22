import sys
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.posteriors.gpytorch import GPyTorchPosterior
import torch
from collections import OrderedDict
from torch.utils.data import (
    TensorDataset, 
    DataLoader
)
"""
Example script to define and train SVGP w/ deep kernel
"""

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, swish):
        if swish:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("swish", Swish()),
            ]))
        else:
            super().__init__(OrderedDict([
                ("fc", torch.nn.Linear(input_dim, output_dim)),
                ("norm", torch.nn.BatchNorm1d(output_dim)),
                ("relu", torch.nn.ReLU(True)),
            ]))


class DenseNetwork(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dims, swish=True):
        prev_dims = [input_dim] + list(hidden_dims[:-1])
        layers = OrderedDict([
            (f"hidden{i + 1}", _LinearBlock(prev_dim, current_dim, swish=swish))
            for i, (prev_dim, current_dim) in enumerate(zip(prev_dims, hidden_dims))
        ])
        self.output_dim = hidden_dims[-1]

        super().__init__(layers)


# gp model with deep kernel
class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256) ):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
        )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.RBFKernel()
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)


def train_gp(model, train_x, train_y, 
    num_epochs=1000, train_batch_size=1024,
    lr=0.01,
    scheduler="multistep", gamma=0.3,
    elbo_beta=1.0,
    mll_type="ELBO",
    device="cpu", tracker=None, 
    save_model=True, save_path=None,
    test_x=None, test_y=None,
    val_x=None, val_y=None,
    load_run_path=None,
    debug=False, verbose=True, save_u=False, obj_name=None,
    lengthscale_only=False, 
    ):


    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    previous_epoch = 0
    if load_run_path is not None:
        print("Loading model ", load_run_path)
        last_run = torch.load(load_run_path)
        model.load_state_dict(last_run["model"])
        previous_epoch = last_run["epoch"]

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    
    optimizer = torch.optim.Adam([ 
        {'params': model.parameters()},   
    ], lr=lr)

    if lengthscale_only:
        optimizer =  torch.optim.Adam([
                {'params': model.covar_module.base_kernel.raw_lengthscale}, 
            ], lr=lr) 
        

    # check_optimizer(optimizer1, name="optimizer1")
    # check_optimizer(optimizer2, name="optimizer2")

    milestones = [int(num_epochs*len(train_loader)/3), int(2*num_epochs*len(train_loader)/3)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
     
    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)

    min_val_rmse = float("Inf")
    min_val_nll = float("Inf")
    model.train()

    for i in range(num_epochs-previous_epoch):
        for k, (x_batch, y_batch) in enumerate(train_loader):
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            optimizer.zero_grad()
            output = model.likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

        means = output.mean.cpu()
        stds  = output.variance.sqrt().cpu()
        rmse = torch.mean((means - y_batch.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch.cpu()).mean()
        if tracker is not None:
            tracker.log({
                "loss": loss.item(), 
                "training_rmse": rmse,
                "training_nll": nll,    
                "ls": model.covar_module.base_kernel.lengthscale.mean().item(),
                # "ls": model.covar_module.lengthscale.mean().item(),
            }, step=i+previous_epoch)
        if i % 10 == 0:
            # print(f"loss: {loss.item()}, lengthscale: {model.covar_module.base_kernel.lengthscale.mean().item()}")
            if val_x is not None:
                min_val_rmse, min_val_nll = val_gp(model, val_x, val_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                    min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                    )
            if test_x is not None:
                _, _, test_rmse, test_nll = eval_gp(model, test_x, test_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
            
            if debug:
                print("u.shape, ", model.variational_strategy.inducing_points.shape)
                print("u.grad, ", model.variational_strategy.inducing_points.grad.abs().mean().item())
                # print("u_covar.grad, ", model.variational_strategy.inducing_points_covar.grad.abs().mean().item())
                print("weight_mean.grad, ", model.feature_extractor_mean.hidden2.fc.weight.grad.abs().mean().item())
                print("weight_covar.grad, ", model.feature_extractor_covar.hidden2.fc.weight.grad.abs().mean().item())
                # print("covar.grad, ", model.variational_strategy._variational_distribution.chol_variational_covar.grad.abs().mean().item())
                # print("mean.grad, ", model.variational_strategy._variational_distribution.variational_mean.grad.abs().mean().item())
                # print("raw_lengthscale.grad, ", model.covar_module.base_kernel.raw_lengthscale.grad.abs().mean().item())
                # print("mean_const.grad, ", model.mean_module.constant.grad.abs().mean().item())
                # print("raw_noise.grad, ", model.likelihood.raw_noise.grad.abs().mean().item())
                sys.stdout.flush()
            if verbose:
                print(f"\n\nEpoch {i}, loss: {loss.item():.3f}, nll: {nll:.3f}, rmse: {rmse:.3e}")
                print(f"testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")
                sys.stdout.flush()

            model.train()
            
            if save_model:
                state = {"model": model.state_dict(), "epoch": i}
                torch.save(state, f'{save_path}.model')

    if val_x is not None:
        min_val_rmse, min_val_nll = val_gp(model, val_x, val_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                    min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                    )
    if test_x is not None:
        _, _, test_rmse, test_nll = eval_gp(model, test_x, test_y,
            test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
        print(f"\nLast testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")
    

    print("model.lengthscale: ", model.covar_module.base_kernel.lengthscale.mean().item())
    # print("model.lengthscale: ", model.covar_module.lengthscale.mean().item())
    # save inducing locations
    if save_u:
        import pickle as pkl
        pkl.dump(model.variational_strategy.inducing_points.detach().cpu(), open(f'./u_svgp_{obj_name}.pkl', 'wb'))
        print(f"Saved u_svgp_{obj_name}")
        
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
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            preds = model.likelihood(model(x_batch))
            # preds = model(x_batch)
            if device == "cuda":
                means = torch.cat([means, preds.mean.cpu()])
                variances = torch.cat([variances, preds.variance.cpu()])
                # print("means = ", preds.mean.cpu()[:10])
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
            preds = model.likelihood(model(x_batch))
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


if __name__ == "__main__":
    # example to load model and trian on random data 
    N = 100
    train_bsz = 10
    n_epochs = 3
    n_inducing = 10
    dim = 32
    train_x = torch.randn(N, dim)
    train_y = torch.randn(N,1)

    # Initialize model: 
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
    """ NOTE: hidden_dims is a tuple giving the number of nodes 
        in each hidden layer in the neural net 
    """
    model = GPModelDKL(
        inducing_points=train_x[0:n_inducing,:].cuda(), 
        likelihood=likelihood,
        hidden_dims=(16, 16) 
    ).cuda()

    # initialize mll: 
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_x.size(-2))

    # train model: 
    model = model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr':0.001} ], lr=0.001)
    train_dataset = TensorDataset(train_x.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for e in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda()).sum() 
            loss.backward()
            optimizer.step()
        print(f"epoch: {e}, loss: {loss.item()}")
    
# Expected output: 
# epoch: 0, loss: 16.277769088745117
# epoch: 1, loss: 20.076805114746094
# epoch: 2, loss: 20.745248794555664
# (loss numbers may vary due to randomness)

