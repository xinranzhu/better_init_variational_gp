import torch 
def levenberg_marquardt(f, J, x, theta, nsteps=100, rtol=1e-8, tau=1e-3):

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
            x = xnew
            theta = theta_new
            fx = fxnew
            Jx = J(x,theta)
            Hx = torch.matmul(Jx.T, Jx)
            # Reset re-scaling parameter, update damping
            mu = mu*max(1.0/3.0, 1.0-2.0*(rho.item()-1.0)**3)
            v = 2.0
        else:
            # Rescale damping5tb
            mu = mu*v
            v = 2*v
    
    return x, theta, norm_hist
    
 