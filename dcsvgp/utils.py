import os
import sys
sys.path.append("./PyTorch-LBFGS/functions")

import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategyDecoupledConditionals

from LBFGS import LBFGS, FullBatchLBFGS

from model import SVGP, DCSVGP


def train(
    model, likelihood, train_x, train_y, test_x, test_y,
    num_epochs=1000, lr=0.01,
    save_loc=None, tracker=None,
):

    # optimizer = FullBatchLBFGS(list(model.parameters()) + list(likelihood.parameters()))
    optimizer = FullBatchLBFGS(model.parameters())
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0), beta=1.)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        return loss

    loss = closure()
    loss.backward()

    for i in range(num_epochs):
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 50}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

        with torch.no_grad():
            train_rmse, train_nll = evaluate(model, likelihood, train_x, train_y)
            test_rmse, test_nll = evaluate(model, likelihood, test_x, test_y)

            if i % 100 == 0:
                print(
                    "iter {:4d},".format(i),
                    "loss {:f},".format(loss.item()),
                    "mean ls {:f},".format(model.variational_strategy.covar_module_mean.lengthscale.item()) if isinstance(model, DCSVGP) else "",
                    "covar ls {:f},".format(model.covar_module.lengthscale.item()),
                    "noise {:f}".format(likelihood.noise.item()),
                    "train rmse {:f}".format(train_rmse),
                    "test rmse {:f}".format(test_rmse),
                    "train nll {:f}".format(train_nll),
                    "test nll {:f}".format(test_nll),
                )

                if save_loc is not None:
                    torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'likelihood_state_dict': likelihood.state_dict(),
                    }, os.path.join(save_loc, "epoch-{:d}.tar".format(i)))
                
            if tracker is not None:
                tracker.log(
                    {
                        'iter': i,
                        'loss': loss.item(),
                        "mean ls": model.variational_strategy.covar_module_mean.lengthscale.item() if isinstance(model, DCSVGP) else model.covar_module.lengthscale.item(),
                        "covar ls": model.covar_module.lengthscale.item(),
                        "noise": likelihood.noise.item(),
                        "train rmse": train_rmse,
                        "test rmse": test_rmse,
                        "train nll": train_nll,
                        "test nll": test_nll,
                    }
                )

    return model, likelihood


def evaluate(model, likelihood, test_x, test_y):
    pred_dist = model(test_x)

    rmse = (pred_dist.mean - test_y).square().mean().sqrt()
    nll = -pred_dist.log_prob(test_y) / test_y.size(0)

    return rmse, nll
