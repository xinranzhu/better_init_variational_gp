import numpy as np
import torch 

class RBFKernel:
    def __init__(self):
        pass
    
class CubicKernel(RBFKernel):
    def __init__(self):
        super(CubicKernel, self).__init__()
    def phi(self, rho):
        return rho**3
    def Drho_phi(self, rho):
        return 3*rho**2
    def Dtheta_phi(self):
        return 0

class SEKernel(RBFKernel):
    def __init__(self, theta):
        super(SEKernel, self).__init__()
        self.theta = theta
    def phi(self, rho):
        s = rho/self.theta
        return torch.exp(-s*s/2)
    def Drho_phi(self, rho):
        s = rho/self.theta
        return -s*torch.exp(-s*s/2)/self.theta
    def Dtheta_phi(self, rho):
        s = rho/self.theta
        return -s*torch.exp(-s*s/2)*(-s/self.theta)
    
        