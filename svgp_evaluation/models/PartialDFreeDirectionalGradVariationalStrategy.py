#!/usr/bin/env python3
# free
from copy import deepcopy
from telnetlib import X3PAD
from termios import VT1
import warnings
import time
import torch
import pickle as pkl
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, TriangularLazyTensor, delazify
from gpytorch.settings import trace_mode
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from gpytorch.utils.warnings import OldVersionWarning
from gpytorch.variational._variational_strategy import _VariationalStrategy


def _ensure_updated_strategy_flag_set(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


class DirectionalGradVariationalStrategy(_VariationalStrategy):
    r"""
    The DfreeDSVGP variational strategy, but with partial directional directional info for inducing points only.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :type learn_inducing_locations: `bool`, optional

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(self, model, inducing_points, inducing_directions, 
        inducing_values_num, variational_distribution, 
        learn_inducing_locations=True,
    ):
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)
        
        self.num_inducing   = len(inducing_points)
        self.inducing_values_num = inducing_values_num
        # get the inducing directions
        self.num_directions = inducing_values_num.max().item()
        # get inducing_values_idx
        inducing_values_idx = []
        for i in range(self.num_inducing):
            cur_num_directions = inducing_values_num[i].item()
            for j in range(self.num_directions):
                idx = i*self.num_directions + j
                if cur_num_directions > 0:
                    inducing_values_idx.append(idx)
                    cur_num_directions -= 1
        self.inducing_values_idx = inducing_values_idx

        # get selected_idx
        # Example of index selection in covar matrix
        # u1, u2-v1, u2, u3-v3
        # full_directions = [0; 0; v1; v2; v3; 0]
        # inducing_values_idx = [2 3 4]
        # inducing_values_num = [0 2 1]
        # selected index in matrix = [0 3 4 5 6 7] 
        selected_idx = []
        idx = 0
        for i in range(self.num_inducing):
            selected_idx.append(idx)
            for j in range(1,1+max(inducing_values_num)):
                idx += 1
                if j <= inducing_values_num[i]:
                    selected_idx.append(idx) 
            idx += 1 
        self.selected_idx = selected_idx

        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.register_parameter(name="inducing_directions", parameter=torch.nn.Parameter(inducing_directions.clone()))
        # self.register_buffer("variational_inducing_directions_initialized", torch.tensor(0))

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).double(), jitter=settings.cholesky_jitter.value())
        return TriangularLazyTensor(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):

        full_inputs = torch.cat([inducing_points,x],dim=-2)
        # predicts mean for each output
        test_mean = self.model.mean_module(x)  


        # inducing_directions contains different # directions for each inducing points (including 0)
        # inducing_values_idx indicates locations of inducing_directions in a full set of directions
        # inducing_values_num indicates # directions for each point
        num_induc = inducing_points.size(-2)
        dim = inducing_points.size(-1)
        full_directions = torch.zeros(num_induc*self.num_directions, dim).to(self.inducing_directions.device)
        full_directions[self.inducing_values_idx] = self.inducing_directions

        kwargs['v1'] = full_directions.to(x.device)
        kwargs['v2'] = None
        # self.model.covar_module.base_kernel.set_num_directions(num_directions,num_derivative_directions)
        self.model.covar_module.set_num_directions(self.num_directions,0)
        full_output = self.model.covar_module(inducing_points,x, **kwargs)[self.selected_idx]
        induc_data_covar  = full_output.evaluate()
       
        # kwargs['v1'] = derivative_directions.to(x.device)
        # kwargs['v2'] = full_directions.to(x.device)
        # self.model.covar_module.set_num_directions(num_derivative_directions,self.num_directions)
        # # self.model.covar_module.base_kernel.set_num_directions(num_directions,num_derivative_directions)
        # full_output = self.model.covar_module(x,inducing_points, **kwargs)[::(num_derivative_directions+1),self.selected_idx]
        # data_induc_covar = full_output.evaluate()
        data_induc_covar = induc_data_covar.T
 

        kwargs['v1'] = full_directions.to(x.device)
        kwargs['v2'] = full_directions.to(x.device)
        # self.model.covar_module.base_kernel.set_num_directions(num_directions,num_derivative_directions)
        self.model.covar_module.set_num_directions(self.num_directions,self.num_directions)
        full_output = self.model.forward(inducing_points, **kwargs)
        # induc_induc_covar  = full_output.lazy_covariance_matrix.add_jitter()
        induc_induc_covar = full_output.lazy_covariance_matrix[self.selected_idx,:][:,self.selected_idx]


        kwargs['v1'] = None
        kwargs['v2'] = None
        # self.model.covar_module.base_kernel.set_num_directions(num_directions,num_derivative_directions)
        self.model.covar_module.set_num_directions(0,0)
        full_output = self.model.forward(x, **kwargs)
        data_data_covar  = full_output.lazy_covariance_matrix

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(full_inputs.dtype)
        # term K_ZZ^{-1/2} K_XZ^T 
        interp_term_trans = L.inv_matmul(data_induc_covar.transpose(-1,-2).double()).to(full_inputs.dtype)
        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        # predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean
        predictive_mean = (interp_term_trans.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean
        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(1e-4).evaluate()
                + interp_term_trans.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
            )
        else:
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term_trans.transpose(-1, -2), middle_term @ interp_term),
            )
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x, prior=False, **kwargs):
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter())

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).double()
                whitened_mean = L.inv_matmul(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate().double()
                whitened_covar = RootLazyTensor(L.inv_matmul(covar_root).to(variational_dist.loc.dtype))
                whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                self._variational_distribution.initialize_variational_distribution(whitened_variational_distribution)

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)
