import torch
import sys
sys.path.append("../")
from splines import spline_K, Dspline_K
from utils import check_cuda_memory

import functorch
from functorch import make_functional_with_buffers
import gpytorch


class ResidualFunctional():
    def __init__(self,
        kernel, m, d,
        outputscale=None, sigma=None
    ):
        self.func, _, self.buffers = make_functional_with_buffers(kernel)
        self.m = m
        self.d = d

        self.outputscale = outputscale
        self.sigma = sigma

    def _residual(self, u, x, y, params, sigma):
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

    def residual(self, inputs, x, y):
        u = inputs[:self.m * self.d].view(self.m, self.d)

        lengthscale = torch.nn.functional.softplus(inputs[-1])

        return self._residual(u, x, y, (lengthscale, self.outputscale), self.sigma)

    def jacobian(self, inputs, x, y):
        return functorch.jacrev(self.residual, argnums=0)(inputs, x, y)


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
    n = 5
    d = 2
    m = 3

    import gpytorch
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    functional = ResidualFunctional(
        kernel,
        m=m,
        d=d,
        sigma=None,
    )

    u = torch.randn(m, d)
    x = torch.randn(n, d)
    y = torch.randn(n)
    # sigma = torch.tensor([1e-2])

    # params = (torch.tensor([0.6]),)

    # residual = functional.residual(u, x, y, params, sigma)
    # print(residual.shape)

    # jacobian = functional.jacobian(u, x, y, params, sigma)
    # print(jacobian.shape)

    lengthscale = torch.tensor([0.6])
    outputscale = torch.tensor([0.7])
    sigma = torch.tensor([1e-2])

    inputs = torch.cat(
        (u.view(-1), lengthscale, outputscale, sigma),
        dim=-1
    )
    residual = functional.residual(inputs, x, y)
    print(residual.shape)

    jacobian = functorch.jacrev(functional.residual, argnums=0)(inputs, x, y)
    print(jacobian.shape)
