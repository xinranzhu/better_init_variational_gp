
import sys
import time
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategySepLengthscale
from torch.utils.data import TensorDataset, DataLoader

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, kernel_type='se', 
        learn_inducing_locations=True, 
        use_ngd=False, 
        ):
        
        if use_ngd:
            variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        else:
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        variational_strategy = VariationalStrategySepLengthscale(self, inducing_points, 
                                                   variational_distribution, 
                                                   learn_inducing_locations=learn_inducing_locations)
        super(GPModel, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # XZ: remove the outside scaling factor for now
        if kernel_type == 'se':
            self.covar_module = gpytorch.kernels.RBFKernel()
            self.covar_module_main = gpytorch.kernels.RBFKernel()
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

def check_optimizer(optimizer, name=None):
    print(f"\n{name}: lr = ", optimizer.param_groups[0]['lr'])
    for group in optimizer.param_groups:
        print("lr = ", group['lr'])
        for param in group['params']:
            print(f"contains param with shape = ", param.shape)

def train_gp(model, train_x, train_y, 
    num_epochs=1000, train_batch_size=1024,
    lr=0.01,
    scheduler=None, gamma=0.3,
    elbo_beta=0.1,
    mll_type="ELBO",
    device="cpu", tracker=None, 
    use_ngd=False, ngd_lr=0.1,
    save_model=True, save_path=None,
    test_x=None, test_y=None,
    val_x=None, val_y=None,
    load_run_path=None,
    lr2=None, gamma2=None,
    debug=False, verbose=True
    ):
    gamma2 = gamma if gamma2 is None else gamma2
    lr2 = lr if lr2 is None else lr2


    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    previous_epoch = 0
    if load_run_path is not None:
        last_run = torch.load(load_run_path)
        model.load_state_dict(last_run["model"])
        previous_epoch = last_run["epoch"]

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    
    assert use_ngd == False
   
    # for the mean, use smaller lr
    optimizer_main = torch.optim.Adam([
                    {'params': model.variational_strategy.inducing_points}, 
                    {'params': model.variational_strategy._variational_distribution.variational_mean},
                    {'params': model.covar_module_main.raw_lengthscale},
                    {'params': model.mean_module.constant}, 
    ], lr=lr)

    # for covariance related params, use normal lr
    optimizer_other = torch.optim.Adam([
                        {'params': model.variational_strategy._variational_distribution.chol_variational_covar},
                        {'params': model.likelihood.raw_noise},
                        {'params': model.covar_module.raw_lengthscale},
    ], lr=lr2)


    # print("model.parameters: ")
    # print(list(model.named_parameters()))
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters()}, # inducing points, mean_const, raw_noise, raw_lengthscale 
    # ], lr=lr)

    check_optimizer(optimizer_main, name="optimizer")
    check_optimizer(optimizer_other, name="optimizer")


    if scheduler == "multistep":
        milestones = [int(num_epochs*len(train_loader)/3), int(2*num_epochs*len(train_loader)/3)]
        scheduler_main = torch.optim.lr_scheduler.MultiStepLR(optimizer_main, milestones, gamma=gamma)
        scheduler_other = torch.optim.lr_scheduler.MultiStepLR(optimizer_other, milestones, gamma=gamma2)
    elif scheduler == None:
        lr_sched = lambda epoch: 1.0
        scheduler_main = torch.optim.lr_scheduler.LambdaLR(optimizer_main, lr_lambda=lr_sched)
        scheduler_other = torch.optim.lr_scheduler.LambdaLR(optimizer_other, lr_lambda=lr_sched)
    elif scheduler == "lambda":
        lr_sched = lambda epoch: 0.8 ** epoch
        scheduler_main = torch.optim.lr_scheduler.LambdaLR(optimizer_main, lr_lambda=lr_sched)
        scheduler_other = torch.optim.lr_scheduler.LambdaLR(optimizer_other, lr_lambda=lr_sched)
    

    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_y.size(0), beta=elbo_beta)

    start = time.time()
    min_val_rmse = float("Inf")
    min_val_nll = float("Inf")
    model.train()


    for i in range(num_epochs-previous_epoch):
        for k, (x_batch, y_batch) in enumerate(train_loader):
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            optimizer_main.zero_grad()
            optimizer_other.zero_grad()
            output = model.likelihood(model(x_batch))
            # output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer_main.step()
            optimizer_other.step()
            scheduler_main.step()
            scheduler_other.step()
            
        means = output.mean.cpu()
        stds  = output.variance.sqrt().cpu()
        rmse = torch.mean((means - y_batch.cpu())**2).sqrt()
        nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch.cpu()).mean()
        if tracker is not None:
            tracker.log({
                "loss": loss.item(), 
                "training_rmse": rmse,
                "training_nll": nll,    
            }, step=i+previous_epoch)
        if i % 10 == 0:
            min_val_rmse, min_val_nll = val_gp(model, val_x, val_y,
                test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                )
            _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,
                test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
            
            if debug:
                print("u.grad, ", model.variational_strategy.inducing_points.grad.abs().mean().item())
                print("covar.grad, ", model.variational_strategy._variational_distribution.chol_variational_covar.grad.abs().mean().item())
                print("mean.grad, ", model.variational_strategy._variational_distribution.variational_mean.grad.abs().mean().item())
                print("raw_lengthscale.grad, ", model.covar_module.raw_lengthscale.grad.abs().mean().item())
                print("raw_lengthscale_main.grad, ", model.covar_module_main.raw_lengthscale.grad.abs().mean().item())
                print("mean_const.grad, ", model.mean_module.constant.grad.abs().mean().item())
                print("raw_noise.grad, ", model.likelihood.raw_noise.grad.abs().mean().item())
                print("lengthscale, ", model.covar_module.lengthscale)
                print("lengthscale_main, ", model.covar_module_main.lengthscale)
                sys.stdout.flush()
            if verbose:
                print(f"\n\nEpoch {i}, loss: {loss.item():.3f}, nll: {nll:.3f}, rmse: {rmse:.3e}")
                print(f"testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")
                sys.stdout.flush()

            model.train()
            
            if save_model:
                state = {"model": model.state_dict(), "epoch": i}
                torch.save(state, f'{save_path}.model')

    end = time.time()
    training_time = end - start
    min_val_rmse, min_val_nll = val_gp(model, val_x, val_y,
                test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                )
    _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,
        test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
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
    test_batch_size=1024, device="cpu", tracker=None, step=0):

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    model.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    start = time.time()
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
