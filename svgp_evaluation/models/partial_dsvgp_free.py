
import sys
import time
import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution
from torch.utils.data import TensorDataset, DataLoader
from rbf_kernel_directional_grad import RBFKernelDirectionalGrad #.RBFKernelDirectionalGrad
from matern_kernel_directional_grad_ard import MaternKernelDirectionalGrad #.MaternKernelDirectionalGrad
#from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy
from PartialDFreeDirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy



class GPModel(ApproximateGP):
    def __init__(self, inducing_points, inducing_directions, 
        inducing_values_num, 
        kernel_type='se', 
        learn_inducing_locations=True, 
        use_ngd=False,
        ):

        self.num_inducing   = len(inducing_points)
        # self.inducing_values_num = inducing_values_num
        # self.num_directions = int(len(inducing_directions)/self.num_inducing) # num directions per point

        num_directional_derivs = sum(inducing_values_num)

        if use_ngd:
        # variational distribution q(u,g)
          variational_distribution = NaturalVariationalDistribution(self.num_inducing + num_directional_derivs)
        else:
          variational_distribution = CholeskyVariationalDistribution(self.num_inducing + num_directional_derivs)
        # variational strategy q(f)
        variational_strategy = DirectionalGradVariationalStrategy(self,
            inducing_points, inducing_directions, inducing_values_num, variational_distribution, learn_inducing_locations=learn_inducing_locations)
        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # XZ: remove the outside scaling factor for now
        if kernel_type == "se":
          self.covar_module = RBFKernelDirectionalGrad()
        elif kernel_type == 'matern1/2':
          self.covar_module = MaternKernelDirectionalGrad(nu=0.5)
        elif kernel_type == 'matern3/2':
          self.covar_module = MaternKernelDirectionalGrad(nu=1.5)  
        elif kernel_type == 'matern5/2':
          self.covar_module = MaternKernelDirectionalGrad(nu=2.5)

    def forward(self, x, **params):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x, **params)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_inducing_points(gp):
    for params in gp.variational_strategy.named_parameters():
        if params[0] == 'inducing_points':
            return params[1].data


def get_inducing_directions(gp):
    for params in gp.variational_strategy.named_parameters():
        if params[0] == 'get_inducing_directions':
            return params[1].data


def train_gp(model, likelihood, train_x, train_y, 
    num_directions,
    num_epochs=1000, 
    train_batch_size=1024,lr=0.01,
    scheduler=None, gamma=0.3,
    elbo_beta=0.1,
    mll_type="ELBO",
    device="cpu", tracker=None,
    use_ngd=False, ngd_lr=0.1,
    save_model=False,
    save_path=None,
  ):
  """Train a Derivative GP with the Directional Derivative
  Variational Inference method

  train_dataset: torch Dataset
  num_inducing: int, number of inducing points
  num_directions: int, number of inducing directions (per inducing point)
  minbatch_size: int, number of data points in a minibatch
  minibatch_dim: int, number of derivative per point in minibatch training
                 WARNING: This must equal num_directions until we complete
                 the PR in GpyTorch.
  num_epochs: int, number of epochs
  inducing_data_initialization: initialize the inducing points as a set of 
      data points. If False, the inducing points are generated on the unit cube
      uniformly, U[0,1]^d.
  learning_rate_hypers, float: initial learning rate for the hyper optimizer
  learning_rate_ngd, float: initial learning rate for the variational optimizer
  use_ngd, bool: use NGD
  use_ciq, bool: use CIQ
  lr_sched, function handle: used in the torch LambdaLR learning rate scheduler. At
      each iteration the initial learning rate is multiplied by the result of 
      this function. The function input is the epoch, i.e. lr_sched(epoch). 
      The function should return a single number. If lr_sched is left as None, 
      the learning rate will be held constant.
  """

  train_dataset = TensorDataset(train_x, train_y)
  dim = len(train_dataset[0][0])
  n_samples = len(train_dataset)
  train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

  if device == "cuda":
    model = model.to(device=torch.device("cuda"))
    likelihood = likelihood.to(device=torch.device("cuda"))

  # training mode
  model.train()
  likelihood.train()

  if use_ngd:
    print("using NGD")
    variational_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=ngd_lr)
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()}, # ind pts, ind directs, mean const, noise, lengthscale
    ], lr=lr)
  else:
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()}, # inducing points, mean_const, raw_noise, raw_lengthscale 
    ], lr=lr)
    variational_optimizer =  torch.optim.Adam([
        {'params': model.variational_parameters()}, 
    ], lr=lr)
     
  
  if scheduler == "multistep":
    num_batches = int(np.ceil(n_samples/train_batch_size))
    milestones = [int(num_batches*num_epochs/3), int(2*num_batches*num_epochs/3)]
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

  total_step=0
  start = time.time()
  for i in range(num_epochs):
    for x_batch, y_batch in train_loader:
      if device == "cuda":
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
      # redo derivative directions b/c batch size is not consistent
      derivative_directions = torch.eye(dim)[:num_directions]
      derivative_directions = derivative_directions.repeat(len(x_batch),1)
      kwargs = {}
      kwargs['derivative_directions'] = derivative_directions.to(x_batch.device)

      hyperparameter_optimizer.zero_grad()
      variational_optimizer.zero_grad()
      output = likelihood(model(x_batch,**kwargs))
      loss = -mll(output, y_batch)
      loss.backward()
      # step optimizers and learning rate schedulers
      hyperparameter_optimizer.step()
      variational_optimizer.step()
      hyperparameter_scheduler.step()
      variational_scheduler.step()
      total_step +=1
      
    means = output.mean[::num_directions+1].cpu()
    stds  = output.variance.sqrt()[::num_directions+1].cpu()
    rmse = torch.mean((means - y_batch[::num_directions+1].cpu())**2).sqrt()
    nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch[::num_directions+1].cpu()).mean()
    if tracker is not None:
      tracker.log({
          "loss": loss.item(), 
          "training_rmse": rmse,
          "training_nll": nll,     
      })
    if i % 100 == 0:
      print(f"Epoch: {i}; total_step: {total_step}, loss: {loss.item()}, nll: {nll}")
      sys.stdout.flush()
      if save_model:
        torch.save(model.state_dict(), f'{save_path}.model')
        torch.save(model.state_dict(), f'{save_path}.likelihood')
        print("Model saved to ", save_path)
      
  end = time.time()
  training_time = end - start
  if tracker is not None:
        tracker.log({
            "training_time":training_time,       
        })
  return model,likelihood, training_time


def eval_gp(model,likelihood,
    test_x, test_y,
    num_directions=1,test_batch_size=1024,
    device='cpu', tracker=None, step=0):

  
  test_dataset = TensorDataset(test_x, test_y)
  test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
  dim = len(test_dataset[0][0])    

  if device == "cuda":
    model = model.to(device=torch.device("cuda"))
    likelihood = likelihood.to(device=torch.device("cuda"))
  model.eval()
  likelihood.eval()
  
  kwargs = {}
  means = torch.tensor([0.])
  variances = torch.tensor([0.])
  start = time.time()
  num_derivative_directions = 1
  with torch.no_grad():
    for x_batch, _ in test_loader:
      # redo derivative directions b/c batch size is not consistent
    #   derivative_directions = torch.eye(dim)[:num_directions]
      derivative_directions = torch.zeros((num_derivative_directions, dim)) 
      for i in range(num_derivative_directions):
          derivative_directions[i,i] = 1
      derivative_directions = derivative_directions.repeat(len(x_batch),1)
      if device == "cuda":
        x_batch = x_batch.cuda()
        derivative_directions = derivative_directions.cuda()

      kwargs['derivative_directions'] = derivative_directions
      # predict
      preds = likelihood(model(x_batch,**kwargs))
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
        "testing_time": testing_time,
    }, step=step)
  print(f"Done testing! rmse: {rmse:.2e}, nll: {nll:.2e}, time: {testing_time:.2e} sec.")
  
  return means, variances, rmse, nll, testing_time