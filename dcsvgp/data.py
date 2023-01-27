import torch
import gpytorch

from model import ExactGP


def get_data(lengthscale, device):
    """
    Generate data from a 1D Gaussian process in [0, 1].
    """
    ground_truth_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    ground_truth_likelihood.noise = 1e-3

    ground_truth_model = ExactGP(None, None, ground_truth_likelihood)
    ground_truth_model.covar_module.lengthscale = lengthscale

    print(ground_truth_model)
    print(
        "lengthscale {:f},".format(ground_truth_model.covar_module.lengthscale.item()),
        "noise {:f}".format(ground_truth_model.likelihood.noise.item()),
    )

    ground_truth_likelihood.to(device).eval()
    ground_truth_model.to(device).eval()

    x = torch.linspace(0., 1., 100, device=device)
    pred_dist = ground_truth_model(x)

    lower, upper = pred_dist.confidence_region()
    y = pred_dist.sample()

    # Train/test split
    num_data = x.size(0)

    train_idx = torch.rand(num_data, device=device) < 0.8
    test_idx = ~train_idx

    train_x = x[train_idx]
    train_y = y[train_idx]

    test_x = x[test_idx]
    test_y = y[test_idx]

    return train_x, train_y, test_x, test_y, ground_truth_model, ground_truth_likelihood


if __name__ == "__main__":
    device = "cuda:0"
    torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lengthscale', type=float)
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    train_x, train_y, test_x, test_y, model, likelihood = get_data(args.lengthscale, device=device)

    torch.save(
        {
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y,
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
        }, args.path,
    )
