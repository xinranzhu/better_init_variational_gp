import torch
import sys
sys.path.append("../")
from splines import spline_K, spline_K_with_outputscale, Dspline_K
from utils import check_cuda_memory

def spline_rproj_all(u, xx, y, theta, outputscale, phi, sigma=1e-3):
    m = u.shape[0]
    Kxu = spline_K_with_outputscale(xx, u, theta, outputscale, phi)

    A = torch.cat([Kxu, sigma*torch.eye(Kxu.shape[1]).to(device=Kxu.device)], dim=0)
    ybar = torch.cat([y, torch.zeros(m).to(device=y.device)])
    c = torch.linalg.lstsq(A, ybar).solution
    r = ybar - torch.matmul(A, c)
    return r


def spline_Jproj_all(u, xx, y, theta, outputscale, phi, Drho_phi, Dtheta_phi, sigma=1e-3):
    """
    Return a tensor of size (n + m, m * d + 3).
    The last three columns follow the order theta, outputscale, noise.
    """
    m, d = u.shape
    n, d = xx.shape
    Kxu = spline_K_with_outputscale(xx, u, theta, outputscale, phi)

    A = torch.cat([Kxu, sigma*torch.eye(Kxu.shape[1]).to(device=Kxu.device)], dim=0)

    ybar = torch.cat([y, torch.zeros(m).to(device=y.device)])
    c = torch.linalg.lstsq(A, ybar).solution
    r = ybar - torch.matmul(A, c)

    JA = Dspline_K(xx, u, theta, Drho_phi, Dtheta_phi) # n * (m * d + m)
    JA = JA * outputscale

    JA_ex = torch.cat([JA, torch.zeros(m, m * d + m).to(device=JA.device)], dim=0)
    z = torch.matmul(JA_ex.T, r)
    JAtr = torch.zeros(m, m*d+1).to(device=JA.device)
    # print("detelting JA_ex, r")
    # check_cuda_memory()
    # del JA_ex, r
    # check_cuda_memory()

    Q1, R1 = torch.linalg.qr(A) # Q=m+n by m, R=m by m

    # compuet JAc n by md+1
    Jac_theta = torch.matmul(JA[:, m*d:], c)
    JAc = torch.cat([JA[:, :m*d], Jac_theta.reshape(-1,1)], dim=1)
    # del Jac_theta

    for j in range(m):
        J = range(j*d,(j+1)*d)
        JAc[:, J] *= c[j]
        JAtr[j, J] = z[J]
    JAtr[:,-1] = z[m*d:]

    # -JAc + Q1 Q1.T JAc - Q1 R1.T\ JAtr
    # method 1
    # JAc2 = torch.cat([JAc, torch.zeros(m, m*d+1).to(device=JA.device)])        
    # temp = torch.matmul(Q1.T, JAc2) - torch.linalg.solve(R1.T, JAtr)
    # res2 = -JAc2 + torch.matmul(Q1, temp)
    
    # method 2
    T1 = torch.linalg.solve(R1.T, JAtr)
    # print("detelting JAtr, R1, z, c")
    # check_cuda_memory()
    # del JAtr, R1, z, c
    # check_cuda_memory()
    T1 -= torch.matmul(Q1[:n].T, JAc)
    res = torch.matmul(Q1, -T1)
    res[:n] -= JAc

    ret = torch.zeros((n + m, m * d + 3))
    ret[:, :m * d + 1] = res

    # gradient w.r.t. outputscale
    # Kxu already has outputscale in it, and thus we want to divide it by outputscale
    JA_scale = torch.cat([Kxu / outputscale, torch.zeros(m, m, device=u.device)], dim=-2)
    ret[:, -2] = -JA_scale @ c + Q1 @ (
        Q1.T @ JA_scale @ c - torch.linalg.solve_triangular(R1.T, JA_scale.T @ r.unsqueeze(-1), upper=False).squeeze()
    )

    # gradient w.r.t. noise
    # tmp = 2 * torch.cholesky_solve(c.unsqueeze(-1), R1, upper=True).squeeze()
    # ret[:-m, -1] = Kxu @ tmp
    # ret[-m:, -1] = -c + tmp
    JA_sigma = torch.cat([torch.zeros(n,  m, device=u.device), torch.eye(m, m, device=u.device)], dim=-2)
    ret[:, -1] = -JA_sigma @ c + Q1 @ (
        Q1.T @ JA_sigma @ c - torch.linalg.solve_triangular(R1.T, JA_sigma.T @ r.unsqueeze(-1), upper=False).squeeze()
    )

    return ret


if __name__ == "__main__":
    from kernels import SEKernel
    from spline_rproj import spline_rproj, spline_Jproj

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    n = 10
    d = 2
    m = 5

    x = torch.randn(n, d)
    u = torch.randn(m, d)
    y = torch.randn(n)

    def phi(rho, theta):
        return SEKernel(theta=theta).phi(rho)

    def Drho_phi(rho, theta):
        return SEKernel(theta=theta).Drho_phi(rho)

    def Dtheta_phi(rho, theta):
        return SEKernel(theta=theta).Dtheta_phi(rho)

    r1 = spline_rproj(
        u, x, y, theta=0.1,
        phi=phi,
        sigma=1e-2,
    )
    r2 = spline_rproj_all(
        u, x, y, theta=0.1, outputscale=1.,
        phi=phi,
        sigma=1e-2
    )
    assert torch.allclose(r1, r2)

    J1 = spline_Jproj(
        u, x, y, theta=1.,
        phi=phi, Drho_phi=Drho_phi, Dtheta_phi=Dtheta_phi,
        sigma=1e-2,
    )
    J2 = spline_Jproj_all(
        u, x, y, theta=1., outputscale=1.,
        phi=phi, Drho_phi=Drho_phi, Dtheta_phi=Dtheta_phi,
        sigma=1e-2,
    )
    assert J1.size() == J2[:, :-2].size()
    assert torch.allclose(J1, J2[:, :-2])


    def finite_difference(theta, outputscale, sigma):
        epsilon = 1e-6

        f_theta_left = spline_rproj_all(
            u, x, y, theta=theta - epsilon, outputscale=outputscale,
            phi=phi,
            sigma=sigma,
        )
        f_theta_right = spline_rproj_all(
            u, x, y, theta=theta + epsilon, outputscale=outputscale,
            phi=phi,
            sigma=sigma,
        )
        d_theta = (f_theta_right - f_theta_left) / 2 / epsilon

        f_outputscale_left = spline_rproj_all(
            u, x, y, theta=theta, outputscale=outputscale - epsilon,
            phi=phi,
            sigma=sigma,
        )
        f_outputscale_right = spline_rproj_all(
            u, x, y, theta=theta, outputscale=outputscale + epsilon,
            phi=phi,
            sigma=sigma,
        )
        d_outputscale = (f_outputscale_right - f_outputscale_left) / 2 / epsilon

        f_sigma_left = spline_rproj_all(
            u, x, y, theta=theta, outputscale=outputscale,
            phi=phi,
            sigma=sigma - epsilon,
        )
        f_sigma_right = spline_rproj_all(
            u, x, y, theta=theta, outputscale=outputscale,
            phi=phi,
            sigma=sigma + epsilon,
        )
        d_sigma = (f_sigma_right - f_sigma_left) / 2 / epsilon

        return d_theta, d_outputscale, d_sigma

    d_theta, d_outputscale, d_sigma = finite_difference(1., 3.5, 1.2)

    J = spline_Jproj_all(
        u, x, y, theta=1., outputscale=3.5,
        phi=phi, Drho_phi=Drho_phi, Dtheta_phi=Dtheta_phi,
        sigma=1.2,
    )
    assert torch.allclose(J[:, -3], d_theta)
    assert torch.allclose(J[:, -2], d_outputscale)
    assert torch.allclose(J[:, -1], d_sigma)
