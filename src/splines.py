import torch
import math
import numpy as np

def spline_K(x, u, theta, phi, V=None):
    m,d = u.shape
    n,d = x.shape
    R = u.reshape(1,m,d)-x.reshape(n,1,d)
    NR = torch.norm(R, dim=2)
    Kxu = phi(NR, theta=theta)
    if V is None: # no directions
        return Kxu
    else:
        # compute Kxu where u has some directions.
        Kxu2 = torch.empty((n, 0)).to(x.device).to(device=x.device)
        for i in range(m):
            ki = Kxu[:,i] # [k1i; k2i; ...; kni], the i-th column of K_xu
            Kxu2 = torch.cat([Kxu2, ki.reshape(-1,1)], dim=1)
            if i in V.keys(): # if ui has directions
                x_ui = x.reshape(n,1,d) - u[i].reshape(1,1,d)
                K_x_ui = ki.reshape(n,1,1) * x_ui # n by 1 by d
                Vi = torch.tensor(V[i]).to(device=x.device) # p by d directions
                KV = (torch.matmul(K_x_ui, Vi.T).squeeze())/theta/theta
                Kxu2 = torch.cat([Kxu2, KV.reshape(n, Vi.shape[0])], dim=1)
        return Kxu2


def spline_K_with_outputscale(x, u, theta, outputscale, phi, V=None):
    return spline_K(x, u, theta, phi, V) * outputscale


def Qxy(x, y, W, D, theta, phi):
    """
    helper function to compute block Qxy, 
    where x and y are two points with directions W and D respectively.
    W = q by d and D = p by d
    Qxy = [ kxy                 kxy/θ^2 (x-y)'d1 ... kxy/θ^2 (x-y)'dp
            -kxy/θ^2 (x-y)'w1   [
            ...                        QH = QH1 + QH2
           -kxy/θ^2 (x-y)'wq                                ]
    ]
    QH1 = -kxy/θ^4 [ w1'(x-y)(x-y)d1 ... 

                                            wq'(x-y)(x-y)dp] = q by p
    QH2 = kxy/θ^2 W * D'
    """
    q, d = W.shape
    p, d = D.shape
    Q = np.zeros((q+1, p+1))
    scale = (phi(torch.norm(x-y), theta=theta)/theta/theta).item()
    Q[0,0] = phi(torch.norm(x-y), theta=theta).item()
    Dxy = D @ (x-y).numpy() # (p,)
    Wxy = W @ (x-y).numpy() # (q,)
    Q[0, 1:] = Dxy * scale
    Q[1:, 0] = -Wxy * scale
    Q[1:, 1:] = (-np.outer(Wxy, Dxy)/theta/theta + (W@D.T)) * scale
    return Q


def spline_Kuu(u, theta, phi, V=None):
    # compute Kuu where u has some directions
    if V is None:
        return spline_K(u, u, theta, phi)

    m, d = u.shape
    l = torch.empty(m).to(dtype=int)
    num_directions = 0
    for i in range(m):
        if i in V.keys():
            p = V[i].shape[0]
            l[i] = p + 1
            num_directions += p
        else:
            l[i] = 1
    # get permutation in descent order
    p = torch.sort(l, descending=True, stable=True)[1]
    # permute u, to group those with directions and those without
    u = u[p]
    num_without_directions = (l[p] == 1).nonzero(as_tuple=True)[0]
    if len(num_without_directions) == 0: # all have directions
        ud = u
        um = None
        Vd = V
    elif len(num_without_directions) == m: # none has directions
        return spline_K(u, u, theta, phi)
    else:
        split_idx = num_without_directions[0]
        um = u[split_idx:]
        ud = u[:split_idx]
        Vd = dict()
        for i in range(len(V)):
            Vd[i] = V[p[i].item()]

    nd = ud.shape[0] # number of points with directions
    Qdd = torch.empty(nd + num_directions, nd + num_directions)
    row_start = 0
    col_start = 0
    for i in range(nd):
        # get col and row range for block i, j
        row_end = row_start + Vd[i].shape[0] + 1
        for j in range(nd):
            col_end = col_start + Vd[j].shape[0] + 1
            Di, Dj = Vd[i], Vd[j]
            Qdd[row_start:row_end, col_start:col_end] = torch.tensor(Qxy(ud[i].cpu(), ud[j].cpu(), Di, Dj, theta, phi))
            # update col_start 
            col_start = col_end
        # update col_start for different rows
        col_start = 0
        # update row_start
        row_start = row_end
    
    Qdd = Qdd.to(device=u.device)
    if um is None:
        return Qdd
    
    Qmd = spline_K(um, ud, theta, phi, V=Vd)
    Qmm = spline_K(um, um, theta, phi)

    # Quu = [Qdd Qmd'; Qmd Qmm]
    Quu1 = torch.cat([Qdd, Qmd.T], dim=1)
    Quu2 = torch.cat([Qmd, Qmm], dim=1)
    Quu = torch.cat([Quu1, Quu2], dim=0)
    
    # get the order back 
    prefix_sum = [1] + [sum(l[p][:i+1]).item() + 1 for i in range(m)][:m-1] 
    r = []
    for i in range(m):
        idx = list(p.numpy()).index(i)
        idx_group = [prefix_sum[idx]+i-1 for i in range(l[i])] 
        r = r + idx_group
    return Quu[r][:,r]



def Dspline_K(x, u, theta, Drho_phi, Dtheta_phi, eps=1e-14):
    m,d = u.shape
    n,d = x.shape
    R = u.reshape(1,m,d)-x.reshape(n,1,d)
    NR = torch.norm(R, dim=2)
    # NR = n by m
    Dphi = Drho_phi(NR, theta=theta)
    mask = NR > eps
    Dphi[mask] = Dphi[mask] / NR[mask]
    Dphi[torch.logical_not(mask)] = 0

    DphiR = Dphi[:,:,None]*R
    del R, Dphi
    DphiR_flat = DphiR.reshape(n, m*d)
    return torch.cat([DphiR_flat, Dtheta_phi(NR, theta=theta)], dim=1)


def spline_eval(x, u, c, theta, phi, V=None):
    Kxu = spline_K(x, u, theta, phi, V=V)
    return torch.matmul(Kxu,c)

def spline_fit(u, x, y, theta, phi, sigma=0., V=None):
    Kxu = spline_K(x, u, theta, phi, V=V)
    m = Kxu.shape[1]
    if sigma == 0.:
        return torch.linalg.lstsq(Kxu,y).solution.cpu()
    else:
        if V is None:
            Kuu = spline_K(u, u, theta, phi)
        else:
            Kuu = spline_Kuu(u, theta, phi, V=V)
        ybar = torch.cat([y, torch.zeros(m).to(device=y.device)])
        jitter =  torch.diag(1e-8*torch.ones(Kuu.shape[0])).to(device=Kuu.device)
        U = torch.linalg.cholesky(Kuu+jitter, upper=True)
        K = torch.cat([Kxu, sigma*U])
        return torch.linalg.lstsq(K,ybar).solution


def rms_vs_truth(u, c, theta, phi, xx_truth, y_truth, V=None):
    u = u.to(device=xx_truth.device)
    c = c.to(device=xx_truth.device)
    y_pred = spline_eval(xx_truth, u, c, theta, phi, V=V)
    num = torch.norm(y_truth - y_pred)
    denorm = math.sqrt(len(y_truth))
    return num/denorm

