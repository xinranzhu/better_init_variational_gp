import sys
import torch 
sys.path.append("../")
from utils import check_cuda_memory

from torch import sigmoid
from torch.nn.functional import softplus
from gpytorch.utils.transforms import inv_softplus


def softplus_scalar(x):
    return 2 * x
    # return softplus(torch.tensor(x)).item()


def sigmoid_scalar(x):
    return 1.
    # return sigmoid(torch.tensor(x)).item()


def inv_softplus_scalar(x):
    return x
    # return inv_softplus(torch.tensor(x)).item()


def levenberg_marquardt_all(f, J, x, theta, outputscale, sigma, nsteps=100, rtol=1e-8, tau=1e-3):
    """
    By default, theta, outputscale, and sigma go through softplus transformation.
    """
    # Evaluate everything at the initial point
    x = torch.clone(x)
    xnew = torch.clone(x)
    theta_new = theta

    fx = f(x, theta, outputscale, sigma)
    Jx = J(x, theta, outputscale, sigma)

    Hx = torch.matmul(Jx.T, Jx)
    
    # print(f"x = {x}, Hx = {Hx}")
    mu = tau * max(torch.diag(Hx)).item()
    v = 2.0
    norm_hist = []
    for k in range(nsteps):
        g = torch.matmul(Jx.T, fx)
        rnorm = torch.norm(g)
        # print(f"k: {k+1}, rnorm: {rnorm}, mu={mu}, v={v}")
        norm_hist.append(rnorm.item())
        if rnorm < rtol:
            return x, theta, outputscale, sigma
        
        # Compute a proposed step and re-evaluate residual vector
        D = torch.diag(mu*torch.ones(Hx.shape[0])).to(device=Hx.device)
        p = torch.linalg.solve((Hx + D), -g)
        
        xnew_flatten = x.flatten() + p[:-3]
        xnew = xnew_flatten.reshape(x.shape)
        theta_new = theta + p[-3].item()
        outputscale_new = outputscale + p[-2].item()
        sigma_new = sigma + p[-1].item()

        fxnew = f(xnew, theta_new, outputscale_new, sigma_new)
        
        # Compute the gain ratio
        rho = (torch.norm(fx)**2 - torch.norm(fxnew)**2) / (torch.norm(fx)**2 - torch.norm(fx+torch.matmul(Jx,p))**2)
        if rho > 0:  # Success!
            # print(f"success, theta_new: {theta_new}")
            # Accept new point
            # print("delete fx, Jx, Hx.")
            # sys.stdout.flush()
            # check_cuda_memory()
            del fx, Jx, Hx
            x = xnew
            theta = theta_new
            outputscale = outputscale_new
            sigma = sigma_new
            fx = fxnew
            del fxnew, xnew, outputscale_new, sigma_new
            Jx = J(x, theta, outputscale, sigma)
            Hx = torch.matmul(Jx.T, Jx)
            # check_cuda_memory()
            # Reset re-scaling parameter, update damping
            mu = mu*max(1.0/3.0, 1.0-2.0*(rho.item()-1.0)**3)
            v = 2.0
        else:
            # Rescale damping5tb
            mu = mu*v
            v = 2*v

    return x, theta, outputscale, sigma, norm_hist
    
 