import torch
import gpytorch

from model import DCSVGP, SVGP
from utils import train
from data import get_data

import wandb

if __name__ == "__main__":
    device = "cuda:0"
    torch.manual_seed(0)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('model', type=str, choices=["svgp", "dcsvgp"])
    parser.add_argument('save_loc', type=str)

    args = parser.parse_args()
    # train_x, train_y, test_x, test_y, model, likelihood = get_data(args.lengthscale, device)
    dataset = torch.load(args.data_path)
    train_x = dataset['train_x']
    train_y = dataset['train_y']
    test_x = dataset['test_x']
    test_y = dataset['test_y']
    # import ipdb; ipdb.set_trace()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(dataset['likelihood_state_dict'])
    likelihood.to(device)

    u0 = torch.randn(100, 1).to(device)

    if args.model == "svgp":
        model = SVGP(inducing_points=u0).to(device)
    elif args.model == "dcsvgp":
        model = DCSVGP(inducing_points=u0).to(device)

    model.train()
    likelihood.train()

    wandb.init(project="1d-gaussian-dcsvgp", name=args.name)

    model, likelihood = train(
        model, likelihood, train_x, train_y, test_x, test_y,
        num_epochs=10000, lr=1.,
        save_loc=args.save_loc,
        tracker=wandb,
    )

    # print(
    #     "mean lengthscale {:f},".format(model.variational_strategy.covar_module_mean.lengthscale.item()),
    #     "covar lengthscale {:f},".format(model.covar_module.lengthscale.item()),
    #     "noise {:f}".format(likelihood.noise.item()),
    # )
