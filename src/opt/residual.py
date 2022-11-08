import sys
sys.path.append("../")

import torch
import functorch
from functorch import make_functional_with_buffers
import gpytorch


class ResidualFunctional():
    def __init__(self,
        kernel, m, d,
        outputscale=None, sigma=None,
        lengthscale_penalty=None
    ):
        self.func, _, self.buffers = make_functional_with_buffers(kernel)
        self.m = m
        self.d = d
        self.lengthscale_penalty = lengthscale_penalty

        self.outputscale = outputscale
        self.sigma = sigma
        self.device = kernel.base_kernel.lengthscale.device

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
            r = ybar - A @ c
            # if self.lengthscale_penalty:
            #     tail = torch.tensor(self.lengthscale_penalty*params[0]).reshape(-1).to(self.device)
            #     res = torch.cat([r, tail])
            #     return res
            # else:
            return r

    def residual(self, inputs, x, y):
        u = inputs[:self.m * self.d].view(self.m, self.d)

        # lengthscale = torch.nn.functional.softplus(inputs[-1])
        lengthscale = inputs[-1]

        return self._residual(u, x, y, (lengthscale, self.outputscale), self.sigma)

    def jacobian(self, inputs, x, y):
        return functorch.jacfwd(self.residual, argnums=0)(inputs, x, y)
        # return functorch.jacrev(self.residual, argnums=0)(inputs, x, y)


if __name__ == "__main__":
    n = 5
    d = 2
    m = 3

    u = torch.randn(m, d)
    x = torch.randn(n, d)
    y = torch.randn(n)

    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    functional = ResidualFunctional(
        kernel, m=m, d=d,
        outputscale=kernel.raw_outputscale, sigma=1e-2,
    )

    inputs = torch.cat(
        (u.view(-1), kernel.base_kernel.raw_lengthscale.view(-1)),
        dim=-1
    )
    residual = functional.residual(inputs, x, y)
    print(residual.shape)

    jacobian = functorch.jacrev(functional.residual, argnums=0)(inputs, x, y)
    print(jacobian.shape)
