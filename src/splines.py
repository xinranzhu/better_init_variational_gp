import torch
import math

def spline_K(x, u, theta, phi, V=None):
    if V is None: # no directions
        m,d = u.shape
        n,d = x.shape
        R = u.reshape(1,m,d)-x.reshape(n,1,d)
        NR = torch.norm(R, dim=2)
        return phi(NR, theta=theta)
    else:
        pass


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
        Kuu = spline_K(u, u, theta, phi, V=V)
        ybar = torch.cat([y, torch.zeros(m).to(device=y.device)])
        jitter =  torch.diag(1e-8*torch.ones(Kuu.shape[0])).to(device=Kuu.device)
        U = torch.linalg.cholesky(Kuu+jitter, upper=True)
        K = torch.cat([Kxu, sigma*U])
        return torch.linalg.lstsq(K,ybar).solution


def rms_vs_truth(u, c, theta, phi, xx_truth, y_truth):
    u = u.to(device=xx_truth.device)
    c = c.to(device=xx_truth.device)
    y_pred = spline_eval(xx_truth, u, c, theta, phi)
    num = torch.norm(y_truth - y_pred)
    denorm = math.sqrt(len(y_truth))
    return num/denorm

