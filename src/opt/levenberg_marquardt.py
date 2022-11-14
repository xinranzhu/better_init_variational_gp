import torch 

import sys
sys.path.append("../")

from kernels import SEKernel
from residual import lstsq_residual, lstsq_jacobian


def phi(rho, theta):
    kernel = SEKernel(theta=theta)
    return kernel.phi(rho)


def Drho_phi(rho, theta):
    kernel = SEKernel(theta=theta)
    return kernel.Drho_phi(rho)


def Dtheta_phi(rho, theta):
    kernel = SEKernel(theta=theta)
    return kernel.Dtheta_phi(rho)


def levenberg_marquardt(
    u, x, y, theta, theta_penalty, sigma,
    nsteps=100, rtol=1e-8, tau=1e-3, output_steps=None, log_each_step=False
):
    m, d = u.shape

    def func_residual(u, theta):
        residual, cache = lstsq_residual(u, x, y, theta, phi, sigma)

        if theta_penalty > 0.:
            return torch.cat([residual, theta_penalty * theta * torch.ones(1, device=u.device)]), cache
        else:
            return residual, cache

    def func_jacobian(u, theta, cache):
        jacobian = lstsq_jacobian(u, x, y, theta, phi, Drho_phi, Dtheta_phi, cache)

        if theta_penalty > 0.:
            t = torch.zeros(jacobian.shape[1], device=u.device)
            t[-1] = theta_penalty
            return torch.cat([jacobian, t.reshape(1, -1)], dim=0)
        else:
            return jacobian

    residual, cache = func_residual(u, theta)
    jacobian = func_jacobian(u, theta, cache)
    hessian = jacobian.T @ jacobian

    # no need to compute hessian explicitly, which is slightly faster
    # mu = tau * jacobian.square().sum(dim=-1).max()
    mu = tau * torch.diag(hessian).max()

    v = 2.0
    norm_hist = []

    res_dict = {}
    if output_steps is not None:
        for step in output_steps:
            res_dict[step] = {}

    log_dict = {
        "u": torch.zeros(nsteps, m, d),
        "grad_norm": torch.zeros(nsteps),
        "theta": torch.zeros(nsteps),
        "training_rmse": torch.zeros(nsteps),
    }

    for k in range(nsteps):
        g = jacobian.T @ residual
        rnorm = torch.norm(g)

        norm_hist.append(rnorm.item())

        if rnorm < rtol:
            if log_each_step:
                return u, theta, norm_hist, res_dict, residual.square().mean().sqrt().cpu(), log_dict
            else:
                return u, theta, norm_hist, res_dict, residual.square().mean().sqrt().cpu()

        # Compute a proposed step and re-evaluate residual vector
        D = mu * torch.eye(hessian.size(0), device=hessian.device)
        p = torch.linalg.solve(hessian + D, -g)

        updated_u = u + p[:-1].view(u.shape)
        updated_theta = theta + p[-1].item()
        updated_residual, cache = func_residual(updated_u, updated_theta)

        # Compute the gain ratio
        rho = (residual.square().sum() - updated_residual.square().sum()) \
            / (residual.square().sum() - (residual + jacobian @ p).square().sum()).item()

        if rho > 0:  # Success!
            # Accept new point
            u = updated_u
            theta = updated_theta

            residual = updated_residual
            jacobian = func_jacobian(u, theta, cache)
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
            res_dict[k] = {
                "u": u.cpu(),
                "theta": theta,
                "train_rmse": residual.square().mean().sqrt().cpu()
            }

        if log_each_step:
            log_dict["u"][k] = u.cpu()
            log_dict["grad_norm"][k] = rnorm.cpu()
            log_dict["theta"][k] = theta
            log_dict["training_rmse"][k] = residual.square().mean().sqrt().cpu()

    if log_each_step:
        return u, theta, norm_hist, res_dict, residual.square().mean().sqrt().cpu(), log_dict
    else:
        return u, theta, norm_hist, res_dict, residual.square().mean().sqrt().cpu()


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cpu"

    # n = 5
    # d = 2
    # m = 3
    n = 11 
    d = 3
    m = 5

    u = torch.randn(m, d, device=device)
    x = torch.randn(n, d, device=device)
    y = torch.randn(n, device=device)

    import time

    start = time.time()
    inducing_points, lengthscale, norm_hist, _, _ = levenberg_marquardt(u, x, y, 1., theta_penalty=10., sigma=1e-2, nsteps=1000)
    end = time.time()

    print(end - start)
    # print(norm_hist)
