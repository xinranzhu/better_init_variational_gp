import sys
import torch 
sys.path.append("../")
# from utils import check_cuda_memory

from spline_rproj import ResidualFunctional


def concatenate(u, lengthscale):
    inputs = u.view(-1)
    inputs = torch.cat((inputs, lengthscale), dim=-1)
    return inputs


def extract(inputs, m, d):
    assert inputs.size(0) == m * d + 1

    u = inputs[:m * d].view(m, d)
    lengthscale = torch.nn.functional.softplus(inputs[-1])

    return u, lengthscale


@torch.no_grad
def levenberg_marquardt(
    u, x, y, kernel, sigma,
    nsteps=100, rtol=1e-8, tau=1e-3):
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

            jacobian = functional.jacobian(u, x, y)
            hessian = jacobian.T @ jacobian

            # Reset re-scaling parameter, update damping
            mu = mu * max(1. / 3., 1. - 2. * (rho - 1.) ** 3)
            v = 2.0
        else:
            # Rescale damping
            mu = mu * v
            v = 2 * v
    
    return extract(inputs, m, d), norm_hist


def _levenberg_marquardt(f, J, x, theta, nsteps=100, rtol=1e-8, tau=1e-3):

    # Evaluate everything at the initial point
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
    
    return x, theta, norm_hist
    
 