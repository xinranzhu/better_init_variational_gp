

import numpy as np
import math
from svgp_utils import fdiff_jac
from torch import is_tensor

class ObjectiveFunction:
  def __init__(self, ef=0, eg=0):
    self.ef = ef
    self.eg = eg
    assert (self.ef >= 0) and (self.eg >= 0)
    self.lb = None
    self.ub = None
    self.dim = None
    self.x0 = None

  def evaluate_function(self, u):
    raise NotImplementedError()
  
  def evaluate_grad(self, u):
    raise NotImplementedError()

  def evaluate_function_noisy(self, u):
    f = self.evaluate_function(u)
    return f + np.random.uniform(-self.ef, self.ef)

  def evaluate_grad_noisy(self, u):
    g = self.evaluate_grad(u)
    return g + np.random.uniform(-self.eg, self.eg, size=self.dim)

  def evaluate_func_grad(self, u):
    # for some NN models, need to implement evaluate_func_grad, b/c f and g have to be evaluated at the same time.
    f = self.evaluate_function(u)
    g = self.evaluate_grad(u)
    return np.concatenate([[f], g])

  def evaluate_func_grad_noisy(self, u):
    f_g = self.evaluate_func_grad(u)
    # f_g[0] = f_g[0] + np.random.uniform(-self.ef, self.ef)
    # f_g[1:] = f_g[1:] + np.random.uniform(-self.eg, self.eg, size=self.dim)
    f_g[0] += np.random.normal(0, self.ef)
    f_g[1:] += np.random.normal(0, self.eg, self.dim)
    return f_g

    
class Branin(ObjectiveFunction):
  r"""Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
  """
  def __init__(self, ef=0, eg=0):
    super().__init__(ef=ef, eg=eg)
    self.fmin = 0.397887
    self.argminx = [np.array([-math.pi, 12.275]),
                    np.array([math.pi, 2.275]),
                    np.array([9.42478, 2.475])]
    self.dim = 2
    self.lb = np.array([-5, 0])
    self.ub = np.array([10, 15])
    

  def evaluate_function(self, u): 
    b = 5.1/(4*math.pi**2)
    c = 5/math.pi
    r = 6
    s = 10
    t = 1/(8*math.pi)
    f = (u[1] - b*u[0]**2 + c*u[0] - r)**2 + s*(1-t)*math.cos(u[0]) + s
    return f 
  
  def evaluate_grad(self, u):
    x1 = u[0]
    x2 = u[1]
    b = 5.1/(4*math.pi**2)
    c = 5/math.pi
    r = 6
    s = 10
    t = 1/(8*math.pi)
    grad = np.array([2*(x2 - b*x1**2 + c*x1 - r)*(-2*b*x1 + c) - s*(1-t)*math.sin(x1),
                     2*(x2 - b*x1**2 + c*x1 - r)]) 
    return grad

class Himmelblau(ObjectiveFunction):
    def __init__(self, ef=0, eg=0, log_eval=False):
        super().__init__(ef=ef, eg=eg)
        self.dim = 2
        self.lb = np.array([-6.0, -6.0])
        self.ub = np.array([6.0, 6.0])
        self.log_eval = log_eval
        
    def evaluate_function(self, u): 
        f = (u[0]**2 + u[1] - 11)**2 + (u[0] + u[1]**2 - 7)**2
        if self.log_eval:
            return np.log(f+10)
        else:
            return f

    def evaluate_grad(self, u):
        x = u[0]
        y = u[1]
        df = [4*x*(x**2 + y - 11)+2*(x + y**2 - 7), 2*(x**2 + y - 11)+4*y*(x + y**2 - 7)]
        f = (u[0]**2 + u[1] - 11)**2 + (u[0] + u[1]**2 - 7)**2
        if self.log_eval:
            return df / (10 + f)
        else:
            return df


class SixHumpCamel(ObjectiveFunction):
  r"""
    dim = 2
    _bounds = [(-3.0, 3.0), (-2.0, 2.0)]
    SixHumpCamel test function.
      (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
            + x1 * x2
            + (4 * x2 ** 2 - 4) * x2 ** 2
     """
  def __init__(self, ef=0, eg=0):
    super().__init__(ef=ef, eg=eg)
    self.fmin = -1.0316
    self.argminx = [np.array([0.0898, -0.7126]),
                    np.array([-0.0898, 0.7126])]
    self.dim = 2
    self.lb = np.array([-3, -2])
    self.ub = np.array([3, 2])

  def evaluate_function(self, u): 
    x1 = u[0]
    x2 = u[1]
    f = (4 - 2.1*x1**2 + (1/3)*x1**4)*(x1**2) + x1*x2 + (-4 + 4*x2**2)*(x2**2)
    return f 
  
  def evaluate_grad(self, u): 
    x1 = u[0]
    x2 = u[1]    
    g1 = (4 - 2.1*x1**2 + (1/3)*x1**4)*2*x1 + (x1**2)*(-2.1*2*x1 + (4/3)*x1**3) + x2
    g2 = x1 + (-4 + 4*x2**2)*2*x2 + (x2**2)*(8*x2)
    return np.array([g1, g2]) 


class StyblinskiTang(ObjectiveFunction):
  r"""StyblinskiTang test function.

    d-dimensional function (usually evaluated on the hypercube [-5, 5]^d):

    H(x) = 0.5 * sum_{i=1}^d (x_i^4 - 16 * x_i^2 + 5 * x_i)

    H has a single global mininimum H(z) = -39.166166 * d at z = [-2.903534]^d
    """
  def __init__(self, ef=0, eg=0, dim=2):
    super().__init__(ef=ef, eg=eg)
    self.fmin =  -39.166166
    self.argminx = -2.903534*np.ones(dim)
    self.dim = dim
    self.lb = -5.*np.ones(dim)
    self.ub = 5.*np.ones(dim)

  def evaluate_function(self, u):
    f = sum([u[i]**4 - 16*(u[i]**2) + 5*u[i] for i in range(self.dim)])
    return f 

  def evaluate_grad(self, u):
    g = np.zeros(self.dim)
    for i in range(self.dim):
      g[i] = 0.5*(4*u[i]**3 - 32*u[i] + 5)
    return g


class SinExp(ObjectiveFunction):
  def __init__(self, ef=0, eg=0):
    super().__init__(ef=ef, eg=eg)
    self.fmin = 0
    self.argminx = None
    self.dim = 2
    self.lb = np.zeros(self.dim)
    self.ub = np.ones(self.dim)

  def evaluate_function(self, u):
    return np.sin(2*np.pi*u[0]**2) + np.exp(u[1]) 

  def evaluate_grad(self, u):
    g1 = 4*np.pi*u[0]*np.cos(2*np.pi*u[0]**2)
    g2 = np.exp(u[1])
    return np.array([g1, g2]) 


class CutestProblem(ObjectiveFunction):
  def __init__(self, problem_name, ef=0, eg=0, **kwargs):
    super().__init__(ef=ef, eg=eg)

    import pycutest
    self.problem_property = pycutest.problem_properties(problem_name)
    # do bound constrained problems for now
    # assert self.problem_property['constraints'] == 'B'
    print(f"Problem {problem_name} has property: {self.problem_property}")

    pycutest.print_available_sif_params(problem_name)
    sifParams = {}
    if 'dim' in kwargs.keys() and self.problem_property['n'] is None:
      sifParams['N'] = kwargs['dim']
    if 'n_constraints' in kwargs.keys() and self.problem_property['m'] is None:
      sifParams['M'] = kwargs['n_constraints']
    print(f"Importing cutest probelm {problem_name} with sifParams: {sifParams}")
    self.problem = pycutest.import_problem(problem_name, sifParams=sifParams)
    assert self.problem_property['degree'] >= 1 

    self.fmin = None
    self.argminx = None
    self.dim = len(self.problem.x0)
    self.x0 = self.problem.x0
    self.lb = self.problem.bl
    self.ub = self.problem.bu

  def evaluate_function(self, u):
    f = self.problem.obj(u)
    return f 

  def evaluate_grad(self, u):
    _, g = self.problem.obj(u, gradient=True)
    return g 
  
  def evaluate_func_grad(self, u):
    f, g = self.problem.obj(u, gradient=True)
    return np.concatenate([ [f], g ])



class Rover(ObjectiveFunction):
  r"""Rover experiment.
  See DSVGP paper for description
  """
  def __init__(self, ef=None, eg=None):
    super().__init__(ef=ef, eg=eg)
    self.fmin = None
    self.argminx = None
    self.dim = 200
    self.lb = -5*np.ones(self.dim)
    self.ub = 5*np.ones(self.dim)    

  def rover_dynamics(self,u,x0):
    m   = 5 # mass
    h   = 0.1 #deltat
    T   = 100 # number of steps
    eta = 1.0 # friction coeff
  
    # state, control
    dim_s = 4 
    dim_c = 2
  
    # dynamics
    A = np.array([[1,0,h,0],[0,1,0,h],[0,0,(1-eta*h/m),0],[0,0,0,(1-eta*h/m)]])
    B = np.array([[0,0],[0,0],[h/m,0],[0,h/m]])
    
    # state control (time is a row)
    x = np.zeros((T,dim_s))
    
    # reshape the control
    u = np.reshape(u,(T,dim_c))
  
    # initial condition
    x[0] = x0
  
    # dynamics
    # x_{t+1}  = Ax_t + Bu_t for t=0,...,T-1
    for t in range(0,T-1):
      x[t+1] = A @ x[t] + B @ u[t]
    return x
  
  def evaluate_function(self, u):
    """
    The rover problem:
    The goal is to learn a controller to drive a rover through four
    waypoints. 
    state: 4dim position, velocity
    control: 2dim x,y forces
  
    input:
    u: length 2T array, open-loop controller
    return:
    cost: float, cost associated with the controller
    """
    if is_tensor(u):
        u = u.detach().cpu().numpy()
    assert len(u) == self.dim
    # initial condition
    x0 = np.array([5,20,0,0])
    # compute dynamics
    x = self.rover_dynamics(u,x0)
    # waypoints
    W = np.array([[8,15,3,-4],[16,7,6,-4],[16,12,-6,-4],[0,0,0,0]])
    way_times = (np.array([10,40,70,100]) - 1).astype(int)
    q1   = 1e0  # penalty on missing waypoint
    q2   = 1e-4 # penalty on control
    # compute cost
    cost = q1*np.sum((x[way_times] - W)**2) + q2*np.sum(u**2)
    return cost
  
  def evaluate_grad(self, u):
    if is_tensor(u):
        u = u.detach().cpu().numpy()
    assert len(u) == self.dim
    """finite difference gradient"""
    return fdiff_jac(self.evaluate_function,u,h=1e-6)

  def evaluate_func_grad(self, u):
    assert len(u) == self.dim
    f = self.evaluate_function(u)
    g = self.evaluate_grad(u)
    return np.concatenate([ [f], g ])
  
