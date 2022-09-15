import torch
from splines import spline_K

# given data points x and existing residual r, 
# compute projected residual with the new points u 
def projected_residual(u, x, theta, phi, r):
    n, d = x.shape
    veck = spline_K(x, u.reshape(1, d), theta, phi).squeeze()
    f = torch.inner(veck, r)
    g = torch.linalg.norm(veck)
    func_val = abs(f/g)
    dk = veck.reshape(n,1) * ((x - u)/theta/theta) # n by d
    grad_val = torch.matmul(dk.T, r)/g - torch.matmul(dk.T, veck) *f/g/g/g
    return func_val, torch.sign(f).item()*grad_val

def gd(u0, obj, obj_grad, lr, nstep):
    f = torch.clone(obj(u0))
    g = torch.clone(obj_grad(u0))
    u = torch.clone(u0)
    for k in range(nstep):
        u = u-lr*g
        g = obj_grad(u)
        f = obj(u)
    return u

def spline_forward_regression(xx, y, kstart, kend, 
    theta, phi, lr=0.1, nstep=10, ncand=100, nsamples=10000, verbose=False):

    if verbose:
        print("spline_forward_regression on going...")

    n, d = xx.shape
    u_cur = torch.clone(xx[:kstart, :])
    
    Kxu = spline_K(xx, u_cur, theta, phi)
    c = torch.linalg.lstsq(Kxu, y).solution
    r = y - torch.matmul(Kxu, c)

    mask = torch.ones(n, dtype=bool)
    mask[:kstart] = False
    idx = torch.arange(n, dtype=int)
    idx_selected = list(range(kstart))
    for k in range(kstart,kend):
        # subsamle_id = torch.randperm(m)[:nsamples]
        def obj(u):
            return -projected_residual(u, xx, theta, phi, r)[0]
        # for continuous version
        def obj_grad(u):
            return -projected_residual(u, xx, theta, phi, r)[1]

        subsamle_id = torch.randperm(xx[mask,:].shape[0])[:min(ncand, xx[mask,:].shape[0])]
        u0_idx = torch.argmin(torch.tensor([obj(u).item() for u in xx[mask,:][subsamle_id, :]])).item()
        u0_idx_original = idx[mask][subsamle_id][u0_idx].item()
        uselect = xx[u0_idx_original, :]
        mask[u0_idx_original] = False
        # uselect = gd(u0, obj, obj_grad, lr, nstep)
        u_cur = torch.cat([u_cur, uselect.reshape(1,-1)])
        idx_selected.append(u0_idx_original)
        # update r 
        Kxu = torch.cat([Kxu, spline_K(xx, uselect.reshape(1, d), theta, phi)], dim=1)
        c = torch.linalg.lstsq(Kxu, y).solution
        r = y - torch.matmul(Kxu, c)
        if verbose:
            print(f"The {k}th point selected at training index {u0_idx_original}.")
    return u_cur



