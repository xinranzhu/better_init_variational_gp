import sys
import torch 
sys.path.append("../")
# from utils import check_cuda_memory

from residual import ResidualFunctional


def concatenate(u, lengthscale):
    inputs = torch.cat((u.view(-1), lengthscale.view(-1)), dim=-1)
    return inputs


def extract(inputs, m, d):
    assert inputs.size(0) == m * d + 1

    u = inputs[:m * d].view(m, d)
    lengthscale = torch.nn.functional.softplus(inputs[-1]).view(1, 1)

    return u, lengthscale


@torch.no_grad()
def levenberg_marquardt(
    u, x, y, kernel, sigma,
    nsteps=100, rtol=1e-8, tau=1e-3,
    output_steps=None, log_each_step=False
):
    """
    Args
        u (m x d tensor): initialization of inducing points
        x (n x d tensor): training data
        y ((n,) tensor): training labels
        kernel (nn.module): the kernel function, e.g. ScaleKernel(RBFKernel())
    """
    m, d = u.size()

    functional = ResidualFunctional(
        kernel, m=m, d=d,
        outputscale=kernel.outputscale, sigma=sigma,
    )

    inputs = concatenate(u, kernel.base_kernel.raw_lengthscale)

    residual = functional.residual(inputs, x, y)
    jacobian = functional.jacobian(inputs, x, y)
    hessian = jacobian.T @ jacobian

    mu = tau * torch.diag(hessian).max()
    v = 2.
    norm_hist = []
    res_dict = {}
    if output_steps is not None:
        for step in output_steps:
            res_dict[step] = {}
    log_dict = {"grad_norm": torch.zeros(nsteps),
                "theta": torch.zeros(nsteps), 
                "training_rmse": torch.zeros(nsteps),
                }

    for k in range(nsteps):
        g = jacobian.T @ residual

        rnorm = torch.norm(g)
        norm_hist.append(rnorm.item())

        if rnorm < rtol:
            return extract(inputs, m, d)
        
        # Compute a proposed step and re-evaluate residual vector
        D = mu * torch.eye(hessian.size(0), device=hessian.device)
        p = torch.linalg.solve(hessian + D, -g) # consider QR decomposition?

        updated_inputs = inputs + p
        updated_residual = functional.residual(updated_inputs, x, y)
        
        # Compute the gain ratio
        rho = (residual.square().sum() - updated_residual.square().sum()) \
            / (residual.square().sum() - (residual + jacobian @ p).square().sum()).item()

        if rho > 0: # Success!
            inputs = updated_inputs
            residual = updated_residual

            jacobian = functional.jacobian(inputs, x, y)
            hessian = jacobian.T @ jacobian

            # Reset re-scaling parameter, update damping
            mu = mu * max(1. / 3., 1. - 2. * (rho - 1.) ** 3)
            v = 2.0
        else:
            # Rescale damping
            mu = mu * v
            v = 2 * v
        if output_steps is not None and k in output_steps:
            print(k)
            sys.stdout.flush()
            res_dict[k] = {"u": x.cpu(), "theta": inputs[-1],
                           "train_rmse": (residual.cpu()**2).mean().sqrt()}

        if log_each_step:
            log_dict["grad_norm"][k] = rnorm.cpu()
            log_dict["theta"][k] = inputs[-1]
            log_dict["training_rmse"][k] = (residual.cpu()**2).mean().sqrt()

    if log_each_step:
        return extract(inputs, m, d), norm_hist, res_dict, (residual.cpu()**2).mean().sqrt(), log_dict
    else:
        return extract(inputs, m, d), norm_hist, res_dict, (residual.cpu()**2).mean().sqrt()

if __name__ == "__main__":
    import gpytorch

    n = 5
    d = 2
    m = 3

    torch.manual_seed(0)
    u = torch.randn(m, d)
    x = torch.randn(n, d)
    y = torch.randn(n)

    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    (inducing_points, lengthscale), norm_hist, _, _ = levenberg_marquardt(u, x, y, kernel, sigma=1e-2)
    print(norm_hist)
