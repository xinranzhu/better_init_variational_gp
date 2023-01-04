import math
import torch
import numpy as np
from botorch.test_functions.synthetic import Branin, SixHumpCamel, StyblinskiTang, Hartmann, SyntheticTestFunction
from torch import Tensor

class Branin_with_deriv(Branin):
    r"""Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
    """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1],1)
        b = 5.1 / (4 * math.pi ** 2)
        c = 5 / math.pi
        t = 1 / (8 * math.pi)
        grad_x2 = 2 * (X[..., 1] - b * X[..., 0] ** 2 + c * X[..., 0] - 6)
        tmp2 = -2 * b * X[..., 0] + c
        tmp3 = - 10 * (1 - t) * torch.sin(X[..., 0])
        grad_x1 = grad_x2 * tmp2 + tmp3
        grad_x1 = grad_x1.reshape(*X.shape[:-1],1)
        grad_x2 = grad_x2.reshape(*X.shape[:-1],1)
        return torch.cat([val, grad_x1, grad_x2], -1)

    def get_bounds(self):
        lb = np.array([item[0] for item in self._bounds])
        ub = np.array([item[1] for item in self._bounds])
        return lb, ub

class SixHumpCamel_with_deriv(SixHumpCamel):
    r"""
    dim = 2
    _bounds = [(-3.0, 3.0), (-2.0, 2.0)]
    SixHumpCamel test function.
      (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
            + x1 * x2
            + (4 * x2 ** 2 - 4) * x2 ** 2
     """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        d = X.shape[-1]     
        x1 = X[..., 0]
        x2 = X[..., 1]
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1],1)
        grad_x1 = x2 + 2 * x1 * (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) + (-4.2 * x1 + 4*x1**3 / 3) * x1 ** 2
        grad_x2 = x1 + 2*x2*(4 * x2 ** 2 - 4) + (8 * x2) * x2 ** 2
        grad_x1 = grad_x1.reshape(*X.shape[:-1],1)
        grad_x2 = grad_x2.reshape(*X.shape[:-1],1)
        return torch.cat([val, grad_x1, grad_x2], -1)

    def get_bounds(self):
            lb = np.array([item[0] for item in self._bounds])
            ub = np.array([item[1] for item in self._bounds])
            return lb, ub


class StyblinskiTang_with_deriv(StyblinskiTang):
    r"""StyblinskiTang test function.

    d-dimensional function (usually evaluated on the hypercube [-5, 5]^d):

    H(x) = 0.5 * sum_{i=1}^d (x_i^4 - 16 * x_i^2 + 5 * x_i)

    H has a single global mininimum H(z) = -39.166166 * d at z = [-2.903534]^d
    """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        d = X.shape[-1]
        #print("d is: ", d)
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1],1) #make last dimension 1
        #print("init val is: ", val)
        for i in range(d):
            cur_grad = 0.5*(4* X[..., i] ** 3 - 32 * X[..., i] + 5)
            # cur_grad = cur_grad.unsqueeze(-1)
            cur_grad = cur_grad.reshape(*X.shape[:-1],1)
            val = torch.cat([val, cur_grad], -1)
        return val

    def get_bounds(self):
        lb = np.array([item[0] for item in self._bounds])
        ub = np.array([item[1] for item in self._bounds])
        return lb, ub


class Hartmann_with_deriv(Hartmann):
    r"""Hartmann synthetic test function.
        Most commonly used is the six-dimensional version (typically evaluated on [0, 1]^6):
        H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )
    """
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        ALPHA = self.ALPHA
        A = self.A 
        P = 1e-4 * self.P
        #print("P: ", P)
        #print("A: ", A)
        #print("ALPHA: ", ALPHA)
        d = X.shape[-1]
        w = X.shape[0]
        assert(d==6)
        #precompute inner summands of H(x): ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 
        exprs = torch.zeros(w, 1)
        for i in range(4):
            cur_expr = torch.zeros(w, 1)
            for j in range(6):
                v = X[..., j].reshape(w, 1)
                cur_expr = cur_expr + A[i, j]*(v - P[i, j])**2
            cur_expr = ALPHA[i]*np.exp(-cur_expr)
            exprs = torch.cat([exprs, cur_expr], 1)
        exprs = exprs[:, 1:]

        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1],1) #make last dimension 1

        #evaluate derivative
        for j in range(d):
            cur_grad = torch.zeros(w, 1)
            for i in range(4):
                v = X[..., j].reshape(w, 1)
                ith = exprs[:, i].reshape(w, 1)
                cur_grad = cur_grad + ith * (-2*A[i, j]*(v-P[i, j]))
            val = torch.cat([val, -cur_grad], 1)
        return val
    
    def get_bounds(self):
        lb = np.array([item[0] for item in self._bounds])
        ub = np.array([item[1] for item in self._bounds])
        return lb, ub


class Welch(SyntheticTestFunction):
    r"""Welch test function (Welch et al 1992). 

    20-dimensional function (usually evaluated on `[-0.5, 0.5]^20`):

        f(x) = 5x_12/(1+x_1) + 5(x_4-x_20)^2 + x_5 + 40x_19^3 - ...
                5x_19 + 0.05x_2 + 0.08x_3 - 0.03x_6 + 0.03x_7 - ...
                0.09x_9 - 0.01x_10 - 0.07x_11 + 0.25x_13^2 - ...
                0.04x_14 + 0.06x_15 - 0.01x_17 - 0.03x_18

    Reference: Einat Neumann Ben-Ari and David M Steinberg. Modeling data from computer experiments:an empirical comparison of kriging with MARS and projection pursuit regression. Quality Engineering, 19(4):327–338, 2007.
    """
    dim = 20
    _bounds = [(-0.5, 0.5) for _ in range(dim)]
    _optimal_value = 0.0
    _optimizers = [(0., 0.)]
    def evaluate_true(self, X: Tensor) -> Tensor:
        term1 = 5*X[..., 11] / (1+X[..., 0])
        term2 = 5 * (X[...,3]-X[..., 19])**2
        term3 = X[..., 4] + 40*X[..., 18]**3 - 5*X[..., 18]
        term4 = 0.05*X[..., 1] + 0.08*X[..., 2] - 0.03*X[..., 5]
        term5 = 0.03*X[..., 6] - 0.09*X[..., 8] - 0.01*X[..., 9]
        term6 = -0.07*X[..., 10] + 0.25*X[..., 12]**2 - 0.04*X[..., 13]
        term7 = 0.06*X[..., 14] - 0.01*X[..., 16] - 0.03*X[..., 17]
        return term1+term2+term3+term4+term5+term6+term7

class Welch_with_deriv(Welch):
    
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        d = X.shape[-1]   
        n = X.shape[:-1]
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        x5 = X[..., 4]
        x6 = X[..., 5]
        x7 = X[..., 6]
        x8 = X[..., 7]
        x9 = X[..., 8]
        x10 = X[..., 9]
        x11 = X[..., 10]
        x12 = X[..., 11]
        x13 = X[..., 12]
        x14 = X[..., 13]
        x15 = X[..., 14]
        x16 = X[..., 15]
        x17 = X[..., 16]
        x18 = X[..., 17]
        x19 = X[..., 18]
        x20 = X[..., 19]
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1], 1)
        grad_x1 = (-5*x12/(1+x1)/(1+x1)).reshape(*X.shape[:-1],1)
        grad_x2 = torch.tensor(0.05).repeat(*X.shape[:-1],1)
        grad_x3 =  torch.tensor(0.08).repeat(*X.shape[:-1],1)
        grad_x4 =  (10*(x4-x20)).reshape(*X.shape[:-1],1)
        grad_x5 =  torch.tensor(1.).repeat(*X.shape[:-1],1)
        grad_x6 =  torch.tensor(-0.03).repeat(*X.shape[:-1],1)
        grad_x7 =  torch.tensor(0.03).repeat(*X.shape[:-1],1)
        grad_x8 =  torch.tensor(0).repeat(*X.shape[:-1],1)
        grad_x9 =  torch.tensor(-0.09).repeat(*X.shape[:-1],1)
        grad_x10 =  torch.tensor(-0.01).repeat(*X.shape[:-1],1)
        grad_x11 =  torch.tensor(-0.07).repeat(*X.shape[:-1],1)
        grad_x12 =  (5/(1+x1)).reshape(*X.shape[:-1],1)
        grad_x13 =  (0.5*x13).reshape(*X.shape[:-1],1)
        grad_x14 =  torch.tensor(-0.04).repeat(*X.shape[:-1],1)
        grad_x15 =  torch.tensor(0.06).repeat(*X.shape[:-1],1)
        grad_x16 =  torch.tensor(0.).repeat(*X.shape[:-1],1)
        grad_x17 =  torch.tensor(-0.01).repeat(*X.shape[:-1],1)
        grad_x18 =  torch.tensor(-0.03).repeat(*X.shape[:-1],1)
        grad_x19 =  (120*x19**2-5).reshape(*X.shape[:-1],1)
        grad_x20 =  (-10*(x4 - x20)).reshape(*X.shape[:-1],1)
        
        return torch.cat([val, grad_x1, grad_x2,
                         grad_x3, grad_x4, grad_x5,
                         grad_x6, grad_x7, grad_x8,
                         grad_x9, grad_x10, grad_x11,
                         grad_x12, grad_x13, grad_x14, 
                         grad_x15, grad_x16, grad_x17,
                         grad_x18, grad_x19, grad_x20], -1)


    def get_bounds(self):
            lb = np.array([item[0] for item in self._bounds])
            ub = np.array([item[1] for item in self._bounds])
            return lb, ub

class Welch2(SyntheticTestFunction):
    r"""A modified Welch test function (Welch et al 1992). 

    20-dimensional function (usually evaluated on `[-0.5, 0.5]^20`):

        f(x) = 5x_12/(1+x_1) + 5(x_4-x_20)^2 + x_5 + 40x_19^3 - ...
                5sin(x_19) + 5cos(x_2) + 8cos(x_3) - 3x_6**4 + 3(x_7**2+x_7) - ...
                9x_9**3 - 10x_10**2 - 7exp(x_11) + 25x_13^2 - ...
                4x_14**2 + 6x_15**2 - 10exp(x_17) - 3exp(x_18)

    Reference: Einat Neumann Ben-Ari and David M Steinberg. Modeling data from computer experiments:an empirical comparison of kriging with MARS and projection pursuit regression. Quality Engineering, 19(4):327–338, 2007.
    """
    dim = 20
    _bounds = [(-0.5, 0.5) for _ in range(dim)]
    _optimal_value = 0.0
    _optimizers = [(0., 0.)]
    def evaluate_true(self, X: Tensor) -> Tensor:
        term1 = 5*X[..., 11] / (1+X[..., 0])
        term2 = 5 * (X[...,3]-X[..., 19])**2
        term3 = X[..., 4] + 40*X[..., 18]**3 - 5*torch.sin(X[..., 18])
        term4 = 5*torch.cos(X[..., 1]) + 8*torch.cos(X[..., 2]) - 3*(X[..., 5]**4)
        term5 = 3*(X[..., 6]**2+X[..., 6]) - 9*(X[..., 8]**3) - 10*(X[..., 9]**2)
        term6 = -7*torch.exp(X[..., 10]) + 25*(X[..., 12]**2) - 4*(X[..., 13]**2)
        term7 = 6*(X[..., 14]**2) - 10*torch.exp(X[..., 16]) - 3*torch.exp(X[..., 17])
        return term1+term2+term3+term4+term5+term6+term7


class Welch2_with_deriv(Welch2):
    
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        d = X.shape[-1]   
        n = X.shape[:-1]
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        x5 = X[..., 4]
        x6 = X[..., 5]
        x7 = X[..., 6]
        x8 = X[..., 7]
        x9 = X[..., 8]
        x10 = X[..., 9]
        x11 = X[..., 10]
        x12 = X[..., 11]
        x13 = X[..., 12]
        x14 = X[..., 13]
        x15 = X[..., 14]
        x16 = X[..., 15]
        x17 = X[..., 16]
        x18 = X[..., 17]
        x19 = X[..., 18]
        x20 = X[..., 19]
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1], 1)
        # f(x) = 5x_12/(1+x_1) + 5(x_4-x_20)^2 + x_5 + 40x_19^3 - ...
        #         5sin(x_19) + 5cos(x_2) + 8cos(x_3) - 3x_6**4 + 3(x_7**2+x_7) - ...
        #         9x_9**3 - 10x_10**2 - 7exp(x_11) + 25x_13^2 - ...
        #         4x_14**2 + 6x_15**2 - 10exp(x_17) - 3exp(x_18)

        grad_x1 = (-5*x12/(1+x1)/(1+x1)).reshape(*X.shape[:-1],1)
        grad_x2 = (-5*torch.sin(x2)).reshape(*X.shape[:-1],1)
        grad_x3 = (-8*torch.sin(x3)).reshape(*X.shape[:-1],1)
        grad_x4 =  (10*(x4-x20)).reshape(*X.shape[:-1],1)
        grad_x5 =  torch.tensor(1.).repeat(*X.shape[:-1],1)
        grad_x6 =  (-12*x6**3).reshape(*X.shape[:-1],1)
        grad_x7 =  (3*(2*x7 + 1.0)).reshape(*X.shape[:-1],1)
        grad_x8 =  torch.tensor(0).repeat(*X.shape[:-1],1)
        grad_x9 =  (-27*x9**2).reshape(*X.shape[:-1],1)
        grad_x10 =  (-20*x10).reshape(*X.shape[:-1],1)
        grad_x11 =  (-7*torch.exp(x11)).reshape(*X.shape[:-1],1)
        grad_x12 =  (5/(1+x1)).reshape(*X.shape[:-1],1)
        grad_x13 =  (50*x13).reshape(*X.shape[:-1],1)
        grad_x14 =  (-8*x14).reshape(*X.shape[:-1],1)
        grad_x15 =  (12*x15).reshape(*X.shape[:-1],1)
        grad_x16 =  torch.tensor(0.).repeat(*X.shape[:-1],1)
        grad_x17 =  (-10*torch.exp(x17)).reshape(*X.shape[:-1],1)
        grad_x18 =  (-3*torch.exp(x18)).reshape(*X.shape[:-1],1)
        grad_x19 =  (120*x19**2-5*torch.cos(x19)).reshape(*X.shape[:-1],1)
        grad_x20 =  (-10*(x4 - x20)).reshape(*X.shape[:-1],1)
        
        return torch.cat([val, grad_x1, grad_x2,
                         grad_x3, grad_x4, grad_x5,
                         grad_x6, grad_x7, grad_x8,
                         grad_x9, grad_x10, grad_x11,
                         grad_x12, grad_x13, grad_x14, 
                         grad_x15, grad_x16, grad_x17,
                         grad_x18, grad_x19, grad_x20], 1)


    def get_bounds(self):
            lb = np.array([item[0] for item in self._bounds])
            ub = np.array([item[1] for item in self._bounds])
            return lb, ub



class SquaredSum(SyntheticTestFunction):
    r"""
        2D function squared sum
        f(x, y) = (x-0.5)**2 + (y-0.5)**2
    """
    dim = 2
    _bounds = [(0, 1) for _ in range(dim)]
    _optimal_value = 0.0
    _optimizers = [(0.5, 0.5)]
    def evaluate_true(self, X: Tensor) -> Tensor:
        fval = (X[..., 0]-0.5)**2 + (X[..., 1]-0.5)**2
        return fval

class SquaredSum_with_deriv(SquaredSum):
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1], 1)
        g1 = 2*(X[..., 0]-0.5).reshape(*X.shape[:-1],1)
        g2 = 2*(X[..., 1]-0.5).reshape(*X.shape[:-1],1)
        return torch.cat([val, g1, g2], 1) 

    def get_bounds(self):
            lb = np.array([item[0] for item in self._bounds])
            ub = np.array([item[1] for item in self._bounds])
            return lb, ub

class Rosenbrock(SyntheticTestFunction):
    r"""
        2D function squared sum
        f(x, y) = (y-x**2)**2 + (x-0.5)**2
    """
    dim = 2
    _bounds = [(0, 1) for _ in range(dim)]
    _optimal_value = 0.0
    _optimizers = [(0.5, 0.5)]
    def evaluate_true(self, X: Tensor) -> Tensor:
        x0 = X[..., 0]*15-5
        x1 = X[..., 1]*15-5
        x0 = (X[..., 0]+5)/15
        x1 = (X[..., 1]+5)/15
        x0 = X[..., 0]*2+5
        x1 = X[..., 1]*2+5
        
        fval = (x1-x0**2)**2 + x0**2
        return fval

class Rosenbrock_with_deriv(Rosenbrock):
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1], 1)
        x0 = X[..., 0]*2+5
        x1 = X[..., 1]*2+5
        gx0 = 2.
        gx1 = 2.
        g1 = -4*(x1-x0**2)*x0*gx0 + 2*x0*gx0
        g2 = 2*(x1-x0**2)*gx1
        g1 = g1.reshape(*X.shape[:-1],1)
        g2 = g2.reshape(*X.shape[:-1],1)
        return torch.cat([val, g1, g2], 1) 

    def get_bounds(self):
            lb = np.array([item[0] for item in self._bounds])
            ub = np.array([item[1] for item in self._bounds])
            return lb, ub


class SinCos(SyntheticTestFunction):
    r"""
        2D function squared sum
        f(x, y) = sin(math.pi * x)**2 + 10*cos(math.pi*x)**2 
    """
    dim = 2
    _bounds = [(0, 1) for _ in range(dim)]
    _optimal_value = 0.0
    _optimizers = [(0.5, 0.5)]
    def evaluate_true(self, X: Tensor) -> Tensor:
        x0 = X[..., 0]
        x1 = X[..., 1]
        # fval = torch.sin(math.pi*x0) + torch.cos(math.pi*x1)
        fval = torch.sin(x0+x1)
        return fval

class SinCos_with_deriv(Rosenbrock):
    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        val = super().evaluate_true(X)
        val = val.reshape(*X.shape[:-1], 1)
        x0 = X[..., 0]
        x1 = X[..., 1]
        gx0 = 1.
        gx1 = 1.
        g1 = torch.cos(x0+x1)
        g2 = -torch.sin(x1)
        g1 = g1.reshape(*X.shape[:-1],1)
        g2 = g2.reshape(*X.shape[:-1],1)
        return torch.cat([val, g1, g2], 1) 

    def get_bounds(self):
            lb = np.array([item[0] for item in self._bounds])
            ub = np.array([item[1] for item in self._bounds])
            return lb, ub