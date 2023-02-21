
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
        use_ngd=False, ard_num_dims=None,
        ):
        assert not use_ngd
        covar_variational_distribution = CholeskyVariationalDistribution(covar_inducing_points.size(-2))
        mean_variational_distribution = gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2))
        covar_variational_strategy = VariationalStrategy(
            self, covar_inducing_points,
            covar_variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )

        print("ard_num_dims = ", ard_num_dims)

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
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern1/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5 ,ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern3/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5 ,ard_num_dims=ard_num_dims)
        elif kernel_type == 'matern5/2':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5 ,ard_num_dims=ard_num_dims)
         
    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
        last_run['model'].pop('variational_strategy.covar_module_mean.raw_lengthscale', None)
        last_run['model'].pop('variational_strategy.covar_module_mean.raw_lengthscale_constraint.lower_bound', None)
        last_run['model'].pop('variational_strategy.covar_module_mean.raw_lengthscale_constraint.upper_bound', None)
        model.load_state_dict(last_run["model"])
        previous_epoch = last_run["epoch"]
        if lengthscale_only:
            hypers = {}
            hypers['covar_module.raw_lengthscale'] =  torch.tensor(0.)
            model.initialize(**hypers)
            previous_epoch = 0

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))

    
    optimizer = torch.optim.Adam([ 
        {'params': model.parameters()},   
    ], lr=lr)

    if lengthscale_only:
        optimizer =  torch.optim.Adam([
                {'params': model.covar_module.raw_lengthscale}, 
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
                "ls": model.covar_module.lengthscale.mean().item(),
            }, step=i+previous_epoch)
        if i % 10 == 0:
            print(f"loss: {loss.item()}, lengthscale: {model.covar_module.lengthscale.mean().item()}")
            if val_x is not None:
                min_val_rmse, min_val_nll = val_gp(model, val_x, val_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                    min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                    )
            if test_x is not None:
                _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,
                    test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
            
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
        _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,
            test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
        print(f"\nLast testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")
    

    print("model.lengthscale: ", model.covar_module.lengthscale.mean().item())
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
