
import sys
import time
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy, OrthogonallyDecoupledVariationalStrategy
from torch.utils.data import TensorDataset, DataLoader


class GPModel(ApproximateGP):
    def __init__(self, mean_inducing_points, covar_inducing_points, kernel_type='se', 
        learn_inducing_locations=True, 
        use_ngd=False,
        ):
        assert not use_ngd
        covar_variational_distribution = CholeskyVariationalDistribution(covar_inducing_points.size(-2))
        mean_variational_distribution = gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2))
        covar_variational_strategy = VariationalStrategy(
            self, covar_inducing_points,
            covar_variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )

        # where predictive_mean = test_mean + Qxu * t, test_mean = Kxz Kzz^{-1} m, 
        # Qxu = Kxu + k_xz K_zz^{-T/2} (S - I) K_zz^{-1/2} K_zu
        variational_strategy = OrthogonallyDecoupledVariationalStrategy(
            covar_variational_strategy, mean_inducing_points, mean_variational_distribution)

        super(GPModel, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # XZ: remove the outside scaling factor for now
        if kernel_type == 'se':
            self.covar_module = gpytorch.kernels.RBFKernel()
            self.covar_module2 = gpytorch.kernels.RBFKernel() # fixed
        elif kernel_type == 'matern1/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        elif kernel_type == 'matern3/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == 'matern5/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
         
    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_inducing_points(gp):
    for params in gp.variational_strategy.named_parameters():
        if params[0] == 'inducing_points':
            return params[1].data

def check_optimizer(optimizer, name=None):
    if optimizer:
        print(f"\n{name}: lr = ", optimizer.param_groups[0]['lr'])
        for group in optimizer.param_groups:
            print("lr = ", group['lr'])
            for param in group['params']:
                print(f"contains param with shape = ", param.shape)

def train_gp(model, train_x, train_y, 
    num_epochs=1000, train_batch_size=1024,
    lr=0.01,
    scheduler=None, gamma=1.0,
    elbo_beta=1.0,
    mll_type="ELBO",
    device="cpu", tracker=None, 
    use_ngd=False, ngd_lr=0.1,
    save_model=True, save_path=None,
    test_x=None, test_y=None,
    val_x=None, val_y=None,
    load_run_path=None,
    debug=False, verbose=True,
    ):

    assert use_ngd==False

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    previous_epoch = 0
    if load_run_path is not None:
        last_run = torch.load(load_run_path)
        model.load_state_dict(last_run["model"])
        previous_epoch = last_run["epoch"]

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    optimizer = torch.optim.Adam([
         {'params': model.parameters()},
    ], lr=lr)
    check_optimizer(optimizer, name="optimizer")



    if scheduler == "multistep":
        milestones = [int(num_epochs*len(train_loader)/3), int(2*num_epochs*len(train_loader)/3)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    elif scheduler == None:
        lr_sched = lambda epoch: 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)
    elif scheduler == "lambda":
        lr_sched = lambda epoch: 0.8 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_sched)

    

    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)
    elif mll_type == "ELBODecoupled":
        mll = gpytorch.mlls.VariationalELBODecoupled(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)
    elif mll_type == "PLLDecoupled":
        mll = gpytorch.mlls.PredictiveLogLikelihoodDecoupled(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)

    start = time.time()
    model.train()

    for i in range(num_epochs-previous_epoch):
        # time1 = time.time()
        for x_batch, y_batch in train_loader:
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
        rmse  = torch.mean((means - y_batch.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch.cpu()).mean()
        if tracker is not None:
            tracker.log({
                "loss": loss.item(), 
                "training_rmse": rmse,
                "training_nll": nll,    
            }, step=i+previous_epoch)
        if i % 10 == 0:
            _, _, _, _= eval_gp(model, val_x, val_y, device=device, 
                tracker=tracker, step=i+previous_epoch, name="val",
                )
            _, _, test_rmse, test_nll = eval_gp(model, test_x, test_y, 
                device=device, tracker=tracker, step=i+previous_epoch, name="testing",
                )
            if debug: 
                print("u.grad, ", model.variational_strategy.inducing_points.grad.abs().mean().item())
                print("z.grad, ", model.variational_strategy.base_variational_strategy.inducing_points.grad.abs().mean().item())
                print("main mean.grad, ", model.variational_strategy._variational_distribution.variational_mean.grad.abs().mean().item())
                print("minor mean.grad, ", model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.grad.abs().mean().item())
                print("minor covar.grad, ", model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.grad.abs().mean().item())
                print("raw_lengthscale.grad, ", model.covar_module.raw_lengthscale.grad.abs().mean().item())
                print("raw_noise.grad, ", model.likelihood.raw_noise.grad.abs().mean().item())
                print("covar: ", model.variational_strategy.base_variational_strategy._variational_distribution().covariance_matrix.diag().mean().item())
            if verbose:
                print(f"Epoch: {i}, loss: {loss.item()}, nll: {nll}, rmse: {rmse}")
                print("\n\n")
            
            model.train()

            if save_model:
                state = {"model": model.state_dict(), "epoch": i}
                torch.save(state, f'{save_path}.model')

    end = time.time()
    training_time = end - start
    _, _, _, _= eval_gp(model, val_x, val_y, device=device, 
                tracker=tracker, step=i+previous_epoch, name="val",
                )
    _, _, test_rmse, test_nll = eval_gp(model, test_x, test_y, device=device, 
        tracker=tracker, step=i+previous_epoch, name="testing",
        )
    print(f"\nLast testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")


    if save_model:
        state = {"model": model.state_dict(), 
            "epoch": i}
        torch.save(state, f'{save_path}.model')

    if tracker is not None:
        tracker.log({
            "training_time":training_time,       
        })
    return model, training_time

def eval_gp(model, test_x, test_y,
    test_batch_size=1024, device="cpu", tracker=None, step=0,
    name="testing"):

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
            f"{name}_rmse":rmse, 
            f"{name}_nll":nll,
        }, step=step)
    return means, variances, rmse, nll


