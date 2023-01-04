
import sys
import random
import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from torch.utils.data import TensorDataset, DataLoader
from rbf_kernel_directional_grad import RBFKernelDirectionalGrad #.RBFKernelDirectionalGrad
from matern_kernel_directional_grad_ard import MaternKernelDirectionalGrad #.MaternKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy



class GPModel(ApproximateGP):
    def __init__(self, inducing_points, inducing_directions, 
        kernel_type='se', 
        learn_inducing_locations=True, 
        **kwargs
        ):

        self.num_inducing   = len(inducing_points)
        self.num_directions = int(len(inducing_directions)/self.num_inducing) # num directions per point
        num_directional_derivs = self.num_directions*self.num_inducing


        # variational distribution q(u,g)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            self.num_inducing + num_directional_derivs)

        # variational strategy q(f)
        variational_strategy = DirectionalGradVariationalStrategy(self,
            inducing_points,inducing_directions,variational_distribution, learn_inducing_locations=learn_inducing_locations)
        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if kernel_type == "se":
          self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernelDirectionalGrad())
        elif kernel_type == 'matern':
          self.covar_module = gpytorch.kernels.ScaleKernel(MaternKernelDirectionalGrad())
   

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

def select_cols_of_y(y_batch,minibatch_dim,dim):
  """
  randomly select columns of y to train on, but always select 
  function values as part of the batch. Otherwise we have
  to keep track of whether we passed in function values or not
  when computing the kernel.

  input
  y_batch: 2D-torch tensor
  minibatch_dim: int, total number of derivative columns of y to sample
  dim: int, problem dimension
  """
  # randomly select columns of y to train on
  idx_y   = random.sample(range(1,dim+1),minibatch_dim) # ensures unique entries
  idx_y  += [0] # append 0 to the list for function values
  idx_y.sort()
  y_batch = y_batch[:,idx_y]

  # dont pass a direction if we load function values
  # E_canonical = torch.eye(dim).to(y_batch.device)
  # derivative_directions = E_canonical[np.array(idx_y[1:])-1]
  derivative_directions = torch.zeros((minibatch_dim, dim))
  for i in range(minibatch_dim):
    y_selected = np.array(idx_y[i+1])-1
    derivative_directions[i, y_selected] = 1
    
  return y_batch,derivative_directions


def train_gp(model, train_x, train_y, 
    num_directions,
    num_epochs=1000, 
    train_batch_size=1024,
    lr=0.01, scheduler="multistep", gamma=1.0,
    mll_type="ELBO",
    device="cpu", 
    tracker=None,
    save_model=False, save_path=None,
    test_x=None, test_y=None,
    load_run_path=None,
    debug=False, verbose=False
  ):
  

  train_dataset = TensorDataset(train_x, train_y)
  dim = len(train_dataset[0][0])
  n_samples = len(train_dataset)
  num_data = (dim+1)*n_samples
  train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

  if device == "cuda":
    model = model.to(device=torch.device("cuda"))


  optimizer = torch.optim.Adam([
      {'params': model.parameters()},
  ], lr=lr)
      
  print("model.params: ", list(model.parameters()))
  milestones = [int(num_epochs*len(train_loader)/3), int(2*num_epochs*len(train_loader)/3)]
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
      
  # mll
  if mll_type=="ELBO":
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=num_data)
  elif mll_type=="PLL": 
    mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=num_data)

  model.train()
  for i in range(num_epochs):
    # loop through minibatches
    for x_batch, y_batch in train_loader:
      if device == torch.device("cuda"):
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
      # select random columns of y_batch to train on
      y_batch,derivative_directions = select_cols_of_y(y_batch,num_directions,dim)
      kwargs = {}
      # repeat the derivative directions for each point in x_batch
      kwargs['derivative_directions'] = derivative_directions.repeat(y_batch.size(0),1)
      # pass in interleaved data... so kernel should also interleave
      y_batch = y_batch.reshape(torch.numel(y_batch))

      optimizer.zero_grad()
      output = model.likelihood(model(x_batch,**kwargs))
      loss = -mll(output, y_batch)
      loss.backward()
      # step optimizers and learning rate schedulers
      optimizer.step()
      scheduler.step()

    means = output.mean[::num_directions+1].cpu()
    stds  = output.variance[::num_directions+1].sqrt().cpu()
    rmse = torch.mean((means - y_batch[::num_directions+1].cpu())**2).sqrt()
    nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch.cpu()).mean()
    if tracker is not None:
      tracker.log({
          "loss": loss.item(), 
          "training_rmse": rmse,
          "training_nll": nll,    
      }, step=i)
    if i % 10 == 0:
      _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,num_directions,
              test_batch_size=1024, device=device, tracker=tracker, step=i)
      model.train()
            
  _, _, test_rmse, test_nll, _ = eval_gp(model, test_x, test_y,num_directions,
    est_batch_size=1024, device=device, tracker=tracker, step=i)
  print(f"\nLast testing rmse: {test_rmse:.3e}, nll:{test_nll:.3f}.")
  return model


def eval_gp(model,
    test_x, test_y,
    test_batch_size=1,
    tracker=None,
    step=None,
    device='cpu'):

  
  test_dataset = TensorDataset(test_x, test_y)
  test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
  dim = len(test_dataset[0][0])    
  model.eval()
  
  kwargs = {}
  means = torch.tensor([0.])
  variances = torch.tensor([0.])
  with torch.no_grad():
    for x_batch, y_batch in test_loader:
      if device == torch.device("cuda"):
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
      # redo derivative directions b/c batch size is not consistent
      derivative_directions = torch.zeros((1, dim)) 
      for i in range(1):
          derivative_directions[i,i] = 1
      derivative_directions = derivative_directions.repeat(len(x_batch),1)
      kwargs['derivative_directions'] = derivative_directions
      # predict
      preds = model.likelihood(model(x_batch,**kwargs))
      if device == torch.device("cuda"):
        means = torch.cat([means, preds.mean[::2].cpu()])
        variances = torch.cat([variances, preds.variance[::2].cpu()])
      else:
        means = torch.cat([means, preds.mean[::2]])
        variances = torch.cat([variances, preds.variance[::2]])

  means = means[1:]
  variances = variances[1:]
  rmse = torch.mean((means - test_y.cpu())**2).sqrt()
  nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y.cpu()).mean()
  if tracker is not None:
    tracker.log({
        "testing_rmse":rmse, 
        "testing_nll":nll,
    }, step=step)
  print("Done Testing!")

  return means, variances, rmse, nll