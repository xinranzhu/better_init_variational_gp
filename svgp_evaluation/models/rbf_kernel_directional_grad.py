#!/usr/bin/env python3

# dw about! 


import torch
import torch.nn.functional as F

from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from gpytorch.kernels.rbf_kernel import RBFKernel, postprocess_rbf


class RBFKernelDirectionalGrad(RBFKernel):
    r"""
    Pass in v1 and v2 through the params. If v1 has n_dir1 directions per
    point in x2 then it should be shape n1*n_dir1 x dim. The directions
    are assumed to be stored in blocks so that the first n_dir1 directions
    belong to x1[0] and the second n_dir1 directions belong to x1[1] etc.
    
    If you have a single set of global directions such as torch.eye(dim), then 
    you can repeat those to make v1 and v2 with 
    v1 = torch.eye(dim).repeat(n1,1)

    Args:
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.


    """

    def forward(self, x1, x2, diag=False, **params):        
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        v1 = params['v1']
        v2 = params['v2']
        if v1 is None and v2 is None:
            return super().forward(x1, x2, diag=diag, **params)
        elif v2 is None: # x2 has no directions
            if diag:
                raise RuntimeError("diag=True only works when x1 == x2 and v1 == v2")
            # number of directions per point
            n_dir1 = int(v1.shape[-2]/n1)
            n_dir2 = 0
            # set num the number of directions for num_outputs_per_input
            self.set_num_directions(n_dir1,n_dir2)
            v1 = F.normalize(v1, dim=1)
            K = torch.zeros(*batch_shape, n1 * (n_dir1 + 1), n2, device=x1.device, dtype=x1.dtype)

            # Scale the inputs by the lengthscale (for stability)
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            # 1) Kernel block
            diff = self.covar_dist(x1_, x2_, square_dist=True, dist_postprocess_func=postprocess_rbf, **params)
            K_11 = diff
            K[..., :n1, :] = K_11

            # 2) Gradient block for x1
            x1_v1 = x1_.reshape(n1,1,d).bmm(torch.transpose(v1.reshape(n1,n_dir1,d),-2,-1))
            x2_v1 = x2_ @ v1.T
            outer  = x1_v1.flatten() - x2_v1
            # outer = [(-y1-x1).dot(V), ..., (-y1+xn2).dot(V);
            #          ... 
            #          (-yn2+x1).dot(V), ..., (-yn1+xn2).dot(V)] = n2 by n1*p

            # permute cols so we get blocks for v1,v2,v3,...
            pi2 = torch.arange(n1 * (n_dir1)).view(n1,n_dir1).t().reshape((n1 * (n_dir1)))
            outer2 = outer[:,pi2]
            # outer2 = [(-y1-X).dot(v1), ..., (-y1+X).dot(vp);
            #          ... 
            #          (-yn2+X).dot(v1), ..., (-yn1+X).dot(vp)] = n2 by n1*p
            outer2  = outer2.t() / self.lengthscale.unsqueeze(-2)
            K[..., n1:, :] = -outer2 * K_11.repeat([n_dir1,*([1] * (n_batch_dims + 1))]) 

            # K = [K(X,Y)       ;
            #      DK(X,Y).dot(V) ] # same directions are grouped together in DK and DDK
            # Apply a perfect shuffle permutation to match the MutiTask ordering
            pi1 = torch.arange(n1 * (n_dir1 + 1)).view(n_dir1 + 1, n1).t().reshape((n1 * (n_dir1 + 1)))
            K = K[..., pi1, :]
            # K = [Q(x1,y1),  ..., Q(x1, yn2);
            #     ... 
            #     Q(xn1,y1), ..., Q(xn1, yn2)],
            # where Q(xi, yj) = [k(xi, yj)           
            #                     dk(xi,yj).dot(v1)   
            #                     ...                 
            #                     dk(xi,yj).dot(vp1) ] = (1+p1) by 1
            return K



        elif v1 is not None and v2 is not None:
            # number of directions per point
            n_dir1 = int(v1.shape[-2]/n1)
            n_dir2 = int(v2.shape[-2]/n2)
            # set num the number of directions for num_outputs_per_input
            self.set_num_directions(n_dir1,n_dir2)

            # normalize directions
            # v1 = (v1.T/torch.norm(v1,dim=1)).T
            # v2 = (v2.T/torch.norm(v2,dim=1)).T
            v1 = F.normalize(v1, dim=1)
            v2 = F.normalize(v2, dim=1)


            # K = torch.zeros(*batch_shape, n1 * (d + 1), n2 * (d + 1), device=x1.device, dtype=x1.dtype)
            K = torch.zeros(*batch_shape, n1 * (n_dir1 + 1), n2 * (n_dir2 + 1), device=x1.device, dtype=x1.dtype)
            if not diag:
                # Scale the inputs by the lengthscale (for stability)
                x1_ = x1.div(self.lengthscale)
                x2_ = x2.div(self.lengthscale)

                # 1) Kernel block
                diff = self.covar_dist(x1_, x2_, square_dist=True, dist_postprocess_func=postprocess_rbf, **params)
                K_11 = diff
                K[..., :n1, :n2] = K_11


                # 2) First gradient block   
                x2_v2 = x2_.reshape(n2,1,d).bmm(torch.transpose(v2.reshape(n2,n_dir2,d),-2,-1))
                # print(f"v1.shape = {v1.shape}, v1 = {v1}")
                # print(f"v2.shape = {v2.shape}, v2 = {v2}")
                # print(f"x1.shape = {x1.shape}, x2.shape = {x2.shape}")
                # v2.shape = n2*n_dir2 by d
                # x1.shape = n1 by d 
                # x1_v2 .shape = n1 by n2*n_dir2
                x1_v2 = x1_ @ v2.T
                outer  = x1_v2 - x2_v2.flatten()
                # let x1 = X = [x1, ..., xn1], x2 = Y = [y1, ..., yn2], 
                # v1 = V = [v1, ..., vp], v2 = W = [w1, ..., wq], 
                # c.dot(W) = [c.dot(w1), ..., c.dot(wq)] = 1 by q
                # outer = [(x1-y1).dot(W), ..., (x1-yn2).dot(W);
                #          ... 
                #          (xn1-y1).dot(W), ..., (xn1-yn2).dot(W)] = n1 by n2*q

                # permute cols so we get blocks for v1,v2,v3,...
                pi1 = torch.arange(n2 * (n_dir2)).view(n2,n_dir2).t().reshape((n2 * (n_dir2)))
                outer1 = outer[:,pi1]/ self.lengthscale.unsqueeze(-2) 
                # outer1 = [(x1-Y).dot(w1), ..., (x1-Y).dot(wq);
                #           ...
                #           (xn1-Y).dot(w1), ..., (xn1-Y).dot(wq)] = n1 by n2*q
                K[..., :n1, n2:] = outer1 * K_11.repeat([*([1] * (n_batch_dims + 1)), n_dir2]) 

                # Second gradient block
                x1_v1 = x1_.reshape(n1,1,d).bmm(torch.transpose(v1.reshape(n1,n_dir1,d),-2,-1))
                x2_v1 = x2_ @ v1.T
                outer  = x1_v1.flatten() - x2_v1
                # outer = [(-y1-x1).dot(V), ..., (-y1+xn2).dot(V);
                #          ... 
                #          (-yn2+x1).dot(V), ..., (-yn1+xn2).dot(V)] = n2 by n1*p

                # permute cols so we get blocks for v1,v2,v3,...
                pi2 = torch.arange(n1 * (n_dir1)).view(n1,n_dir1).t().reshape((n1 * (n_dir1)))
                outer2 = outer[:,pi2]
                # outer2 = [(-y1-X).dot(v1), ..., (-y1+X).dot(vp);
                #          ... 
                #          (-yn2+X).dot(v1), ..., (-yn1+X).dot(vp)] = n2 by n1*p
                outer2  = outer2.t() / self.lengthscale.unsqueeze(-2)
                K[..., n1:, :n2] = -outer2 * K_11.repeat([n_dir1,*([1] * (n_batch_dims + 1))]) 


                # 4) Hessian block (n1*n_dir1, n2*n_dir2)
                outer3 = outer1.repeat(1, n_dir1, 1) * outer2.repeat(1,1,n_dir2)  
                # kronecker product term
                kp = v1 @ v2.T / self.lengthscale.pow(2)
                kp = kp[:,pi1][pi2,:]
                chain_rule = kp - outer3
                K[..., n1:, n2:] = chain_rule * K_11.repeat([*([1] * n_batch_dims), n_dir1,n_dir2])
            
                # K = [K(X,Y)         DK(X,Y).dot(W);
                #      DK(X,Y).dot(V) V.dot(DDK(X,Y).dot(W))] # same directions are grouped together in DK and DDK
                # Apply a perfect shuffle permutation to match the MutiTask ordering
                pi1 = torch.arange(n1 * (n_dir1 + 1)).view(n_dir1 + 1, n1).t().reshape((n1 * (n_dir1 + 1)))
                pi2 = torch.arange(n2 * (n_dir2 + 1)).view(n_dir2 + 1, n2).t().reshape((n2 * (n_dir2 + 1)))
                K = K[..., pi1, :][..., :, pi2]
                # K = [Q(x1,y1),  ..., Q(x1, yn2);
                #     ... 
                #     Q(xn1,y1), ..., Q(xn1, yn2)],
                # where Q(xi, yj) = [k(xi, yj)           dk(xi,yj).dot(w1) ...  dk(xi,yj).dot(wp2)
                #                     dk(xi,yj).dot(v1)   v1'*ddk(xi,yj)*w1 .... v1'*ddk(xi,yj)*wp2
                #                     ...                 ...
                #                     dk(xi,yj).dot(vp1)  vp1'*ddk(xi,yj)*w1 .... vp1'*ddk(xi,yj)*wp2] = (1+p1) by (1+p2)
                return K

            else:
                if not (n1 == n2 and torch.eq(x1, x2).all() and n_dir1 == n_dir2 and torch.eq(v1, v2).all()):
                    raise RuntimeError("diag=True only works when x1 == x2 and v1 == v2")

                kernel_diag = super(RBFKernelDirectionalGrad, self).forward(x1, x2, diag=True)
                grad_diag = torch.ones(*batch_shape, n2, n_dir2, device=x1.device, dtype=x1.dtype) / self.lengthscale.pow(2)
                grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * n_dir2)
                k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
                pi = torch.arange(n2 * (n_dir2 + 1)).view(n_dir2 + 1, n2).t().reshape((n2 * (n_dir2 + 1)))
                return k_diag[..., pi]

    def set_num_directions(self,n_dir1,n_dir2):
        """needed num_outputs_per_intput doesnt take v1,v2 as 
           args"""
        self.n_dir1 = n_dir1
        self.n_dir2 = n_dir2
        # print(f"set_num_directions, n_dir1={n_dir1}, n_dir2={n_dir2}")

    def num_outputs_per_input(self, x1, x2):
        return (self.n_dir1 +1,self.n_dir2 +1)
        # return self.n_dir1+1



if __name__ == '__main__':

  torch.manual_seed(0)
  # generate training data
  n1   = 100
  n2   = n1
  dim = 2
  train_x  = torch.rand(n1,dim)
  # train_x2 = torch.rand(n2,dim)
  train_x2 = train_x
  # set directions
  n_directions = 2
  # v1 = torch.eye(dim)[:n_directions]
  v1 = torch.rand(n_directions,dim)
  v1 = v1.repeat(n1,1)
  # v2 = torch.eye(dim)[:n_directions]
  # v2 = torch.rand(n_directions,dim)
  # v2 = v2.repeat(n2,1)
  # v2 = v1
  v2 = torch.rand(1,dim)
  v2 = v2.repeat(n1,1)

  v1 = (v1.T/torch.norm(v1,dim=1)).T
  v2 = (v2.T/torch.norm(v2,dim=1)).T

  k = RBFKernelDirectionalGrad()
  params = {'v1':v1,'v2':v2}
  K = k(train_x,train_x2, **params)
  print(K.detach().numpy().shape)

  # torch.cholesky(K.add_jitter().evaluate())
  # verify against RBFKernelGrad
  # from gpytorch.kernels import RBFKernelGrad
  # kk = RBFKernelGrad()
  # KK = kk(train_x,train_x2)
  # print(KK.detach().numpy() - K.detach().numpy())
