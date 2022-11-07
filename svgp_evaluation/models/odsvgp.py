
import sys
import time
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy, OrthogonallyDecoupledVariationalStrategy
from torch.utils.data import TensorDataset, DataLoader

def _pivoted_cholesky_init(
    train_inputs,
    kernel_matrix,
    max_length,
    epsilon=1e-6,
):
    r"""
    A pivoted cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. Code is a PyTorch version from [chen2018dpp]_, copied from
    https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py.

    Args:
        train_inputs: training inputs (of shape n x d)
        kernel_matrix: kernel matrix on the training
            inputs
        max_length: number of inducing points to initialize
        epsilon: numerical jitter for stability.

    Returns:
        max_length x d tensor of the training inputs corresponding to the top
        max_length pivots of the training kernel matrix
    """
    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration
    # TODO: use gpytorch's pivoted cholesky instead once that gets an exposed list
    # TODO: ensure this works in batch mode, which it does not currently.
    NEG_INF = -(torch.tensor(float("inf")))
    item_size = kernel_matrix.shape[-2]
    cis = torch.zeros(
        (max_length, item_size), device=kernel_matrix.device, dtype=kernel_matrix.dtype
    )
    di2s = kernel_matrix.diag()
    selected_items = []
    selected_item = torch.argmax(di2s)
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis.pow(2.0)
        di2s[selected_item] = NEG_INF
        selected_item = torch.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    ind_points = train_inputs[torch.stack(selected_items)]

    return ind_points

def _select_inducing_points(
    inputs,
    covar_module,
    num_inducing,
    input_batch_shape,
):
    r"""
    Utility function that evaluates a kernel at given inputs and selects inducing point
    locations based on the pivoted Cholesky heuristic.

    Args:
        inputs: A (*batch_shape, n, d)-dim input data tensor.
        covar_module: GPyTorch Module returning a LazyTensor kernel matrix.
        num_inducing: The maximun number (m) of inducing points (m <= n).
        input_batch_shape: The non-task-related batch shape.

    Returns:
        A (*batch_shape, m, d)-dim tensor of inducing point locations.
    """

    train_train_kernel = covar_module(inputs).evaluate_kernel()

    # base case
    if train_train_kernel.ndimension() == 2:
        inducing_points = _pivoted_cholesky_init(
            train_inputs=inputs,
            kernel_matrix=train_train_kernel,
            max_length=num_inducing,
        )
    # multi-task case
    elif train_train_kernel.ndimension() == 3 and len(input_batch_shape) == 0:
        input_element = inputs[0] if inputs.ndimension() == 3 else inputs
        kernel_element = train_train_kernel[0]
        inducing_points = _pivoted_cholesky_init(
            train_inputs=input_element,
            kernel_matrix=kernel_element,
            max_length=num_inducing,
        )
    # batched input cases
    else:
        batched_inputs = (
            inputs.expand(*input_batch_shape, -1, -1)
            if inputs.ndimension() == 2
            else inputs
        )
        reshaped_inputs = batched_inputs.flatten(end_dim=-3)
        inducing_points = []
        for input_element in reshaped_inputs:
            # the extra kernel evals are a little wasteful but make it
            # easier to infer the task batch size
            kernel_element = covar_module(input_element).evaluate_kernel()
            # handle extra task batch dimension
            kernel_element = (
                kernel_element[0]
                if kernel_element.ndimension() == 3
                else kernel_element
            )
            inducing_points.append(
                _pivoted_cholesky_init(
                    train_inputs=input_element,
                    kernel_matrix=kernel_element,
                    max_length=num_inducing,
                )
            )
        inducing_points = torch.stack(inducing_points).view(
            *input_batch_shape, num_inducing, -1
        )

    return inducing_points

class GPModel(ApproximateGP):
    def __init__(self, mean_inducing_points, covar_inducing_points, kernel_type='se', 
        learn_inducing_locations=True, 
        use_ngd=False,
        ):
        assert not use_ngd
        covar_variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2))
        mean_variational_distribution = gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2))
        covar_variational_strategy = gpytorch.variational.VariationalStrategy(
            self, covar_inducing_points,
            covar_variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )
        variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
            covar_variational_strategy, mean_inducing_points, mean_variational_distribution)

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

def check_optimizer(optimizer, name=None):
    if optimizer:
        print(f"\n{name}: lr = ", optimizer.param_groups[0]['lr'])
        for group in optimizer.param_groups:
            print("lr = ", group['lr'])
            for param in group['params']:
                print(f"contains param with shape = ", param.shape)

def train_gp(model, likelihood, train_x, train_y, 
    num_epochs=1000, train_batch_size=1024,
    learn_inducing_values=True, lr=0.01,
    scheduler=None, gamma=0.3,
    elbo_beta=0.1,
    mll_type="ELBO",
    device="cpu", tracker=None, 
    use_ngd=False, ngd_lr=0.1,
    save_model=True, save_path=None,
    test_x=None, test_y=None,
    val_x=None, val_y=None,
    load_run_path=None,
    learn_main=True, learn_other=True,
    lr2=None, gamma2=None,
    ):
    gamma2 = gamma if gamma2 is None else gamma2
    lr2 = lr if lr2 is None else lr2

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    previous_epoch = 0
    if load_run_path is not None:
        last_run = torch.load(load_run_path)
        model.load_state_dict(last_run["model"])
        likelihood = model.likelihood
        previous_epoch = last_run["epoch"]

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))
        likelihood = likelihood.to(device=torch.device("cuda"))
    
    model.train()
    likelihood.train()

    assert use_ngd==False

    optimizer_main = None
    scheduler_main = None
    scheduler_other = None
    if learn_main:
        optimizer_main = torch.optim.Adam([
                    {'params': model.variational_strategy.inducing_points}, 
                    {'params': model.variational_strategy._variational_distribution.variational_mean},
                ], lr=lr)
    if learn_other:
        optimizer_other = torch.optim.Adam([
                        {'params': model.variational_strategy.base_variational_strategy.parameters()}, # minor inducing, cholesky m and S 
                        {'params': model.mean_module.constant}, 
                        {'params': likelihood.raw_noise},
                        {'params': model.covar_module.raw_lengthscale},
        ], lr=lr2)

    check_optimizer(optimizer_main, name="optimizer_main")
    check_optimizer(optimizer_other, name="optimizer_other")


    if scheduler == "multistep":
        milestones = [int(num_epochs*len(train_loader)/3), int(2*num_epochs*len(train_loader)/3)]
        if optimizer_main is not None:
            scheduler_main = torch.optim.lr_scheduler.MultiStepLR(optimizer_main, milestones, gamma=gamma)
        scheduler_other = torch.optim.lr_scheduler.MultiStepLR(optimizer_other, milestones, gamma=gamma2)
    elif scheduler == None:
        lr_sched = lambda epoch: 1.0
        if optimizer_main is not None:
            scheduler_main = torch.optim.lr_scheduler.LambdaLR(optimizer_main, lr_lambda=lr_sched)
        scheduler_other = torch.optim.lr_scheduler.LambdaLR(optimizer_other, lr_lambda=lr_sched)
    elif scheduler == "lambda":
        lr_sched = lambda epoch: 0.8 ** epoch
        if optimizer_main is not None:
            scheduler_main = torch.optim.lr_scheduler.LambdaLR(optimizer_main, lr_lambda=lr_sched)
        scheduler_other = torch.optim.lr_scheduler.LambdaLR(optimizer_other, lr_lambda=lr_sched)
    
    

    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0), beta=elbo_beta)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.size(0), beta=elbo_beta)

    start = time.time()
    min_val_rmse = float("Inf")
    min_val_nll = float("Inf")
    for i in range(num_epochs-previous_epoch):
        # time1 = time.time()
        for x_batch, y_batch in train_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            if optimizer_main is not None:
                optimizer_main.zero_grad()
            if optimizer_other is not None:
                optimizer_other.zero_grad()
            output = likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            loss.backward()
            if optimizer_main is not None:
                optimizer_main.step()
                scheduler_main.step()
            if optimizer_other is not None:
                optimizer_other.step()
                scheduler_other.step()

        if i == 0:
            # make sure there's grad
            print("u.grad, ", model.variational_strategy.inducing_points.grad.abs().mean().item())
            print("z.grad, ", model.variational_strategy.base_variational_strategy.inducing_points.grad.abs().mean().item())
            print("main mean.grad, ", model.variational_strategy._variational_distribution.variational_mean.grad.abs().mean().item())
            print("minor mean.grad, ", model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.grad.abs().mean().item())
            print("minor covar.grad, ", model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.grad.abs().mean().item())
            print("raw_lengthscale.grad, ", model.covar_module.raw_lengthscale.grad.abs().mean().item())
            print("raw_noise.grad, ", model.likelihood.raw_noise.grad.abs().mean().item())

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
            min_val_rmse, min_val_nll = val_gp(model, likelihood, val_x, val_y,
                test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                )
            _, _, _, _, _ = eval_gp(model, likelihood, test_x, test_y,
                test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
            model.train()
            likelihood.train()
            print(f"Epoch: {i}, loss: {loss.item()}, nll: {nll}, rmse: {rmse}")
            sys.stdout.flush()
            if save_model:
                state = {"model": model.state_dict(), "epoch": i}
                torch.save(state, f'{save_path}.model')
    end = time.time()
    training_time = end - start
    min_val_rmse, min_val_nll = val_gp(model, likelihood, val_x, val_y,
                test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch, 
                min_val_rmse=min_val_rmse, min_val_nll=min_val_nll
                )
    _, _, _, _, _ = eval_gp(model, likelihood, test_x, test_y,
        test_batch_size=1024, device=device, tracker=tracker, step=i+previous_epoch)
    sys.stdout.flush()
    if save_model:
        state = {"model": model.state_dict(), "epoch": i}
        torch.save(state, f'{save_path}.model')

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
        for x_batch, y_batch in test_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
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




def val_gp(model, likelihood, val_x, val_y,
    test_batch_size=1024, device="cpu", tracker=None, step=0,
    min_val_rmse=None, min_val_nll=None):

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

    if device == "cuda":
        model = model.to(device=torch.device("cuda"))
        likelihood = likelihood.to(device=torch.device("cuda"))

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])

    with torch.no_grad():
        for x_batch, _ in val_loader:
            if device == "cuda":
                x_batch = x_batch.cuda()
            preds = likelihood(model(x_batch))
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
