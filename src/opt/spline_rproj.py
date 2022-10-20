import torch
import sys
sys.path.append("../")
from splines import spline_K, Dspline_K
from utils import check_cuda_memory

import functorch
from functorch import make_functional_with_buffers
import gpytorch


class KernelFunctional():
    def __init__(self, kernel):
        self.func, _, self.buffers = make_functional_with_buffers(kernel)

    def residual(self, u, x, y, params, sigma):
        """
        y is of size (n,)
        """
        with gpytorch.settings.trace_mode(), gpytorch.settings.lazily_evaluate_kernels(False):
            m = u.size(0)

            func_nl = lambda params, buffers, x1, x2: self.func(params, buffers, x1, x2).evaluate()

            Kxu = func_nl(params, self.buffers, x, u)
            A = torch.cat(
                [Kxu, sigma * torch.eye(m, device=u.device)],
                dim=-2,
            )
            ybar = torch.cat([y, y.new_zeros(m)], dim=-1)
            c = torch.linalg.lstsq(A, ybar.unsqueeze(-1), rcond=None).solution.squeeze()
            return ybar - A @ c

    def jacobian(self, u, x, y, params, sigma):
        # res_func = lambda u, params, sigma: self.residual(u, x, y, params, sigma)
        return functorch.jacrev(self.residual, argnums=(0, 3, 4))(u, x, y, params, sigma)


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
    # check_cuda_memory()
    del JA_ex, r
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
    del JAtr, R1, z, c
    # check_cuda_memory()
    T1 -= torch.matmul(Q1[:n].T, JAc)
    res = torch.matmul(Q1, -T1)
    res[:n] -= JAc

    return res


if __name__ == "__main__":
    import gpytorch
    functional = KernelFunctional(gpytorch.kernels.RBFKernel())

    u = torch.randn(3, 2)
    x = torch.randn(5, 2)
    y = torch.randn(5)
    sigma = torch.tensor([1e-2])

    params = (torch.tensor([0.6]),)

    residual = functional.residual(u, x, y, params, sigma)
    print(residual.shape)

    jacobian = functional.jacobian(u, x, y, params, sigma)
    print(jacobian[0].shape, jacobian[1][0].shape, jacobian[2].shape)
