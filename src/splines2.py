import torch
import math
import numpy as np

def spline_K(x, u, kernel):
    Kxu = kernel(x,u).evaluate()
    return Kxu


def spline_eval(x, u, c, kernel):
    Kxu = spline_K(x, u, kernel).evaluate()
    return torch.matmul(Kxu,c)

def spline_fit(u, x, y, kernel, sigma=0., ):
    Kxu = spline_K(x, u, kernel).evaluate()
    m = Kxu.shape[1]
    if sigma == 0.:
        return torch.linalg.lstsq(Kxu,y).solution.cpu()
    else:
        Kuu = spline_K(u, u, kernel)
        ybar = torch.cat([y, torch.zeros(m).to(device=y.device)])
        jitter =  torch.diag(1e-8*torch.ones(Kuu.shape[0])).to(device=Kuu.device)
        U = torch.linalg.cholesky(Kuu+jitter, upper=True)
        K = torch.cat([Kxu, sigma*U])
        return torch.linalg.lstsq(K,ybar).solution


def rms_vs_truth(u, c, kernel, xx_truth, y_truth):
    u = u.to(device=xx_truth.device)
    c = c.to(device=xx_truth.device)
    y_pred = spline_eval(xx_truth, u, c, kernel)
    num = torch.norm(y_truth - y_pred)
    denorm = math.sqrt(len(y_truth))
    return num/denorm

