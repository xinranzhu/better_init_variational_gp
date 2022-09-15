
import sys
import time
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy
from torch.utils.data import TensorDataset, DataLoader

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, kernel_type='se', 
        learn_inducing_locations=True, 
        inducing_values_prior=None,
        use_ngd=False,
        ):
        if use_ngd:
            variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        else:
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        # variational_strategy = UnwhitenedVariationalStrategy(self, inducing_points, 
        #                                            variational_distribution, 
        #                                            learn_inducing_locations=learn_inducing_locations)
        variational_strategy = VariationalStrategy(self, inducing_points, 
                                                   variational_distribution, 
                                                   learn_inducing_locations=learn_inducing_locations,
                                                   inducing_values_prior=inducing_values_prior)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # XZ: remove the outside scaling factor for now
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

def get_inducing_points(gp):
    for params in gp.variational_strategy.named_parameters():
        if params[0] == 'inducing_points':
            return params[1].data


def train_gp(model, likelihood, train_x, train_y, 
    num_epochs=1000, train_batch_size=1024,
    learn_inducing_values=True, lr=0.01, 
    scheduler=None, gamma=0.3,
    elbo_beta=0.1,
    mll_type="ELBO",
    device="cpu", tracker=None, 
    use_ngd=False, ngd_lr=0.1,
    ):
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))
        likelihood = likelihood.to(device=torch.device("cuda"))

    model.train()
    likelihood.train()

    if use_ngd:
        print("using NGD")
        variational_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=ngd_lr)
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': model.hyperparameters()},
        ], lr=lr)
    else:
        if learn_inducing_values:
            hyperparameter_optimizer = torch.optim.Adam([
                {'params': model.hyperparameters()}, # inducing points, mean_const, raw_noise, raw_lengthscale 
            ], lr=lr)
            variational_optimizer =  torch.optim.Adam([
                {'params': model.variational_parameters()}, 
            ], lr=lr)
        else:
            hyperparameter_optimizer = torch.optim.Adam([
                {'params': model.hyperparameters()}, # inducing points, mean_const, raw_noise, raw_lengthscale 
            ], lr=lr)
            # don't learn variational mean, make sure the inducing points are fixed as well
            assert 'variational_strategy.inducing_points' not in [param[0] for param in model.named_hyperparameters()]
            variational_optimizer = torch.optim.Adam([
                {'params': model.variational_strategy._variational_distribution.chol_variational_covar},
            ], lr=lr)

    if scheduler == "multistep":
        milestones = [int(num_epochs/3), int(2*num_epochs/3)]
        hyperparameter_scheduler = torch.optim.lr_scheduler.MultiStepLR(hyperparameter_optimizer, milestones, gamma=gamma)
        variational_scheduler = torch.optim.lr_scheduler.MultiStepLR(variational_optimizer, milestones, gamma=gamma)
    elif scheduler == None:
        lr_sched = lambda epoch: 1.0
        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)
    elif scheduler == "lambda":
        lr_sched = lambda epoch: 0.8 ** epoch
        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(hyperparameter_optimizer, lr_lambda=lr_sched)
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(variational_optimizer, lr_lambda=lr_sched)
    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0), beta=elbo_beta)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.size(0))

    start = time.time()
    for i in range(num_epochs):
        for x_batch, y_batch in train_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            hyperparameter_optimizer.zero_grad()
            variational_optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            hyperparameter_optimizer.step()
            variational_optimizer.step()
            hyperparameter_scheduler.step()
            variational_scheduler.step()

        means = output.mean.cpu()
        stds  = output.variance.sqrt().cpu()
        rmse = torch.mean((means - y_batch.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch.cpu()).mean()
        if tracker is not None:
            tracker.log({
                "loss": loss.item(), 
                "training_rmse": rmse,
                "training_nll": nll,     
            })
        if i % 100 == 0:
            print(f"Epoch: {i}, loss: {loss.item()}, nll: {nll}, rmse: {rmse}")
            sys.stdout.flush()
    end = time.time()
    training_time = end - start
    if tracker is not None:
        tracker.log({
            "training_time":training_time,       
        })
    return model, likelihood, training_time

def eval_gp(model, likelihood, test_x, test_y,
    test_batch_size=1024, device="cpu", tracker=None, step=0):

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))
        likelihood = likelihood.to(device=torch.device("cuda"))

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    start = time.time()
    with torch.no_grad():
        for x_batch, _ in test_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
            preds = likelihood(model(x_batch))
            if device == "cuda":
                means = torch.cat([means, preds.mean.cpu()])
                variances = torch.cat([variances, preds.variance.cpu()])
            else:
                means = torch.cat([means, preds.mean])
                variances = torch.cat([variances, preds.variance])
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
            "testing_time": testing_time 
        }, step=step)
    return means, variances, rmse, nll, testing_time
