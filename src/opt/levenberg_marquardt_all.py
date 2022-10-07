import sys
import torch 
sys.path.append("../")
from utils import check_cuda_memory

from torch.nn.functional import softplus, sigmoid
from gpytorch.utils.transforms import inv_softplus


def levenberg_marquardt_all(f, J, x, theta, outputscale, sigma, nsteps=100, rtol=1e-8, tau=1e-3):
    """
    By default, theta, outputscale, and sigma go through softplus transformation.
    """
    if transform:
        theta = inv_softplus(theta)
        outputscale = inv_softplus(outputscale)
        sigma = inv_softplus(sigma)

    # Evaluate everything at the initial point
    x = torch.clone(x)
    xnew = torch.clone(x)
    theta_new = theta

    if transform:
        fx = f(x, softplus(theta), softplus(outputscale), softplus(sigma))
        Jx = J(x, softplus(theta), softplus(outputscale), softplus(sigma))
        Jx[:, -3] *= sigmoid(theta)
        Jx[:, -2] *= sigmoid(outputscale)
        Jx[:, -1] *= sigmoid(sigma)
    else:
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
            return x, theta
        
        # Compute a proposed step and re-evaluate residual vector
        D = torch.diag(mu*torch.ones(Hx.shape[0])).to(device=Hx.device)
        p = torch.linalg.solve((Hx + D), -g)
        
        xnew_flatten = x.flatten() + p[:-3]
        xnew = xnew_flatten.reshape(x.shape)
        theta_new = theta + p[-3].item()
        outputscale_new = outputscale + p[-2].item()
        sigma_new = sigma + p[-1].item()

        if transform:
            fxnew = f(xnew, softplus(theta_new), softplus(outputscale_new), softplus(sigma_new))
        else:    
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
            if transform:
                Jx = J(x, softplus(theta), softplus(outputscale), softplus(sigma))
                Jx[:, -3] *= sigmoid(theta)
                Jx[:, -2] *= sigmoid(outputscale)
                Jx[:, -1] *= sigmoid(sigma)
            else:
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
    
 