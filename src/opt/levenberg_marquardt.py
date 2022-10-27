import sys
import torch 
sys.path.append("../")
from utils import check_cuda_memory

def levenberg_marquardt(f, J, x, theta, nsteps=100, rtol=1e-8, tau=1e-3, output_steps=None, log_each_step=False):

    # Evaluate everything at the initial point
    m, dim = x.shape
    x = torch.clone(x)
    xnew = torch.clone(x)
    theta_new = theta
    fx = f(x,theta)
    Jx = J(x,theta)
    Hx = torch.matmul(Jx.T, Jx)
    
#     print(f"x = {x}, Hx = {Hx}")
    mu = tau * max(torch.diag(Hx)).item()
    v = 2.0
    norm_hist = []

    res_dict = {}
    if output_steps is not None:
        for step in output_steps:
            res_dict[step] = {}

    log_dict = {"u": torch.zeros(nsteps, m, dim),
                "grad_norm": torch.zeros(nsteps),
                "theta": torch.zeros(nsteps), 
                "training_rmse": torch.zeros(nsteps),
                }
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
        
        xnew_flatten = x.flatten() + p[:-1]
        xnew = xnew_flatten.reshape(x.shape)
        theta_new = theta + p[-1].item()
        fxnew = f(xnew, theta_new)
        
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
            fx = fxnew
            del fxnew, xnew
            Jx = J(x,theta)
            Hx = torch.matmul(Jx.T, Jx)
            # check_cuda_memory()
            # Reset re-scaling parameter, update damping
            mu = mu*max(1.0/3.0, 1.0-2.0*(rho.item()-1.0)**3)
            v = 2.0
        else:
            # Rescale damping5tb
            mu = mu*v
            v = 2*v

        if output_steps is not None and k in output_steps:
            print(k)
            sys.stdout.flush()
            res_dict[k] = {"u": x.cpu(), "theta": theta,
                           "train_rmse": (fx.cpu()**2).mean().sqrt()}

        if log_each_step:
            log_dict["u"][k] = x.cpu()
            log_dict["grad_norm"][k] = rnorm.cpu()
            log_dict["theta"][k] = theta
            log_dict["training_rmse"][k] = (fx.cpu()**2).mean().sqrt()

    if log_each_step:
        return x, theta, norm_hist, res_dict, (fx.cpu()**2).mean().sqrt(), log_dict
    else:
        return x, theta, norm_hist, res_dict, (fx.cpu()**2).mean().sqrt()
    
 