import torch

import sys
sys.path.append("../")

from splines import spline_K, Dspline_K


def lstsq_residual(u, x, y, theta, phi, sigma=1e-3):
    m, _ = u.shape

    Kxu = spline_K(x, u, theta, phi)

    A = torch.cat([Kxu, sigma * torch.eye(m, device=u.device)], dim=0)
    ybar = torch.cat([y, torch.zeros(m, device=u.device)])

    # Q is of size (n + m, m) and R is of size (m, m)
    Q, R = torch.linalg.qr(A)

    # compute the residual
    # the parenthesis is crucial (computing Q @ Q.T first is slower)
    # r = ybar - Q @ (Q.T @ ybar)

    # if both c and r are need, then this is more efficient
    c = torch.linalg.solve_triangular(R, Q.T @ ybar.unsqueeze(-1), upper=True).squeeze()
    r = ybar - A @ c

    return r, c, Q, R


def lstsq_jacobian(u, x, y, theta, Drho_phi, Dtheta_phi, r, c, Q, R, sigma=1e-3):
    m, d = u.shape
    n, d = x.shape

    JA = Dspline_K(x, u, theta, Drho_phi, Dtheta_phi)  # n*md+m
    JA_ex = torch.cat([JA, torch.zeros(m, m * d + m).to(device=JA.device)], dim=0)
    z = torch.matmul(JA_ex.T, r)
    JAtr = torch.zeros(m, m * d + 1).to(device=JA.device)

    # compuet JAc n by md+1
    Jac_theta = torch.matmul(JA[:, m*d:], c)
    JAc = torch.cat([JA[:, :m*d], Jac_theta.reshape(-1, 1)], dim=1)

    for j in range(m):
        J = range(j * d, (j+1)*d)
        JAc[:, J] *= c[j]
        JAtr[j, J] = z[J]
    JAtr[:, -1] = z[m*d:]

    T1 = torch.linalg.solve_triangular(R.T, JAtr, upper=False)
    T1 -= Q[:n].T @ JAc
    res = Q @ -T1
    res[:n] -= JAc

    return res
