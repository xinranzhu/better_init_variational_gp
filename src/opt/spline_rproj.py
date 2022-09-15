import torch
import sys
sys.path.append("../")
from splines import spline_K, Dspline_K

def spline_rproj(u, xx, y, theta, phi, sigma=1e-3):
    m = u.shape[0]
    Kxu = spline_K(xx, u, theta, phi)
    A = torch.cat([Kxu, sigma*torch.eye(Kxu.shape[1]).to(device=Kxu.device)], dim=0)
    ybar = torch.cat([y, torch.zeros(m).to(device=y.device)])
    c = torch.linalg.lstsq(A, ybar).solution
    r = ybar - torch.matmul(A, c)
    return r

def spline_Jproj(u, xx, y, theta, phi, Drho_phi, Dtheta_phi, sigma=1e-3):
    m, d = u.shape
    n, d = xx.shape
    Kxu = spline_K(xx, u, theta, phi)

    # m+n by m
    A = torch.cat([Kxu, sigma*torch.eye(Kxu.shape[1]).to(device=Kxu.device)], dim=0)

    ybar = torch.cat([y, torch.zeros(m).to(device=y.device)])
    c = torch.linalg.lstsq(A, ybar).solution
    r = ybar - torch.matmul(A, c)
    JA = Dspline_K(xx, u, theta, Drho_phi, Dtheta_phi) # n*md+m
    JA_ex = torch.cat([JA, torch.zeros(m, m*d+m).to(device=JA.device)], dim=0)
    z = torch.matmul(JA_ex.T, r)
    JAtr = torch.zeros(m, m*d+1).to(device=JA.device)

    Q1, R1 = torch.linalg.qr(A) # Q=m+n by m, R=m by m

    # compuet JAc n by md+1
    Jac_theta = torch.matmul(JA[:, m*d:], c)
    JAc = torch.cat([JA[:, :m*d], Jac_theta.reshape(-1,1)], dim=1)
    for j in range(m):
        J = range(j*d,(j+1)*d)
        JAc[:, J] *= c[j]
        JAtr[j, J] = z[J]
    JAc = torch.cat([JAc, torch.zeros(m, m*d+1).to(device=JA.device)])        
    JAtr[:,-1] = z[m*d:]

    # -JAc + Q1 Q1.T JAc - Q1 R1.T\ JAtr
    temp = torch.matmul(Q1.T, JAc) - torch.linalg.solve(R1.T, JAtr)
    res = -JAc + torch.matmul(Q1, temp)
    return res

