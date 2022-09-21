import torch
import sys
from splines import spline_K
from sklearn.cluster import KMeans

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
    theta, phi, lr=0.1, nstep=10, 
    ncand=100, nsamples=10000, 
    max_obj_tol=1e-6, count_random_max=10,
    device="cpu", verbose=False):
    torch.set_default_dtype(torch.float64) 

    if device == "cuda":
        xx = xx.cuda()
        y = y.cuda()

    n, d = xx.shape
    u_cur = torch.clone(xx[:kstart, :])
    
    Kxu = spline_K(xx, u_cur, theta, phi)
    c = torch.linalg.lstsq(Kxu, y).solution
    r = y - torch.matmul(Kxu, c)

    mask = torch.ones(n, dtype=bool)
    mask[:kstart] = False
    idx = torch.arange(n, dtype=int)
    idx_selected = list(range(kstart))
    count_random = 0
    for k in range(kstart,kend):
        if count_random > count_random_max:
            if verbose:
                print("select the rest by kmeans")
            res = KMeans(n_clusters=kend-k).fit(xx[mask].cpu().numpy())
            ufwd2 = torch.tensor(res.cluster_centers_, device=u_cur.device)
            return torch.cat([u_cur, ufwd2], dim=0)

        # subsamle_id = torch.randperm(m)[:nsamples]
        def obj(u):
            return projected_residual(u, xx, theta, phi, r)[0]
        def obj_grad(u):
            return projected_residual(u, xx, theta, phi, r)[1]

        m2 = xx[mask,:].shape[0]
        subsamle_id = torch.randperm(m2)[:ncand]
        obj_vals = torch.tensor([obj(u).item() for u in xx[mask,:][subsamle_id, :]])
        if max(obj_vals) < max_obj_tol:
            if verbose:
                print("select randomly")
            count_random += 1
            u0_idx = torch.randperm(m2)[0]
            u0_idx_original = idx[mask][u0_idx]
        else:
            u0_idx = torch.argmax(obj_vals).item()
            u0_idx_original = idx[mask][subsamle_id][u0_idx].item()

        # obj_vals = torch.tensor([obj(u).item() for u in xx[mask,:]])
        # u0_idx = torch.argmax(obj_vals).item()
        # u0_idx_original = idx[mask][u0_idx].item()

        uselect = xx[u0_idx_original, :]
        mask[u0_idx_original] = False
        # uselect = gd(u0, obj, obj_grad, lr, nstep)
        u_cur = torch.cat([u_cur, uselect.reshape(1,-1)])
        idx_selected.append(u0_idx_original)
        # update r 
        Kxu = torch.cat([Kxu, spline_K(xx, uselect.reshape(1, d), theta, phi)], dim=1)
        c = torch.linalg.lstsq(Kxu, y).solution
        r = y - torch.matmul(Kxu, c)
        if k % 50 == 0:
            print(f"k={k}")
            sys.stdout.flush()
        if verbose:
            print(f"k={k}, u0_idx_original={u0_idx_original}, max_obj={max(obj_vals)}")
        # print(f"obj_vals = ", torch.sort(obj_vals, descending=True)[0][:5])
        # print(f"opt idx = ", torch.sort(obj_vals, descending=True)[1][:5])
        # print("singular value of Kxu: ", torch.svd(Kxu).S)
    return u_cur



