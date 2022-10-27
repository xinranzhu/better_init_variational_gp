# baseline GP regression, use all training data 
import numpy as np
import time
import sys
import random
import torch
import gpytorch
import pickle as pkl
sys.path.append("./models")
from svgp import GPModel, get_inducing_points, train_gp, eval_gp, _select_inducing_points
from svgp_utils import meshgrid_uniform
from eval_experiment import Experiment


try:
    import fire
except ModuleNotFoundError:
    print("Optional dependencies for experiments not installed.")

try:
    import wandb
    LOG_WANDB = True
except ModuleNotFoundError:
    LOG_WANDB = False

class SVGP_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="SVGP")
        
    def init_hypers(self, num_inducing=2, 
        init_method="random", 
        init_expid="-",
        learn_u=True, 
        learn_m=True,
        use_ngd=False, ngd_lr=0.1,
        save_model=False,
        init_theta=True,
        init_noise=True,
        init_covar=True,
        init_mean=True, 
        lm_step=None,
        ):

        self.method_args['init_hypers'] = locals()
        # m is the size of Kuu, recorded to maintain the same computational cost
        m = num_inducing
        self.method_args['init_hypers']['m'] = m
        del self.method_args['init_hypers']['self']

        self.learn_u = learn_u
        self.learn_m = learn_m

        if init_method.startswith("random"):
            rand_index = random.sample(range(self.train_n), num_inducing)
            u0 = self.train_x[rand_index, :]
        elif init_method == "kmeans":
            from sklearn.cluster import KMeans
            xk = self.train_x.cpu().numpy()
            kmeans = KMeans(n_clusters=num_inducing, random_state=0).fit(xk)
            u0 = kmeans.cluster_centers_
        else: # method = "fwd", "lm" or "tr_newton"
            if init_method == "lm_init":
                res = pkl.load(open(f"../results/{self.obj_name}-{self.dim}_kmeans_m{m}_{init_expid}_{self.seed}.pkl", "rb"))
            else:
                if lm_step is None:
                    res = pkl.load(open(f"../results/{self.obj_name}-{self.dim}_{init_method}_m{m}_{init_expid}_{self.seed}.pkl", "rb"))
                else:
                    res = pkl.load(open(f"../results/{self.obj_name}-{self.dim}_{init_method}_m{m}_{init_expid}_{self.seed}_step{lm_step}.pkl", "rb"))
            u0 = res["u"].to(device=self.device)
            c = res["c"]
            sigma = res["sigma"]
            theta = res["theta"]
            time_cost = res["time"]
            if init_covar:
                Sbar = res["Sbar"]
                try:
                    Lbar = torch.linalg.cholesky(Sbar)
                except: # failed to initialize variational covariance
                    init_covar=False 
            print(f"Pretraining by {init_method} cost: {time_cost} sec.")
            assert u0.shape[0] == num_inducing and u0.shape[1] == self.dim
        
        u0 = torch.tensor(u0)
        model = GPModel(inducing_points=u0, 
                learn_inducing_locations=learn_u,
                use_ngd=use_ngd)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        if init_method == "pivchol":
            # compute pivoted cholesky initialization for inducing points
            input_batch_shape = self.train_x.shape[:-2]
            u0_new = _select_inducing_points(
                self.train_x,
                model.covar_module,
                num_inducing,
                input_batch_shape,
            )
            print("norm difference between u0 and u0_new: ", torch.norm(u0-u0_new))
            model = GPModel(inducing_points=u0_new, 
                learn_inducing_locations=learn_u,
                use_ngd=use_ngd)

        if init_method not in {"random", "kmeans", "random_init_noise", "pivchol"}:
            hypers = {}
            hypers_likelihood = {}
            if use_ngd:
                hypers["variational_strategy._variational_distribution.natural_vec"] = c.to(u0.device)
            else:
                if init_theta:
                    print("initializing theta.")
                    hypers['covar_module.lengthscale'] =  torch.tensor(theta)
                if init_noise: 
                    print("initializing noise.")
                    hypers_likelihood["likelihood.noise_covar.noise"] = torch.tensor(sigma**2)
                if init_covar:
                    print("initializing covar.")
                    hypers["variational_strategy._variational_distribution.chol_variational_covar"] = Lbar.to(u0.device)
                    model.variational_strategy.variational_covar_initialized = torch.tensor(1)
                if init_mean:
                    print("initializing mean.")
                    hypers["variational_strategy._variational_distribution.variational_mean"] = c.to(u0.device)
                    model.variational_strategy.variational_mean_initialized = torch.tensor(1)
            model.initialize(**hypers)
            likelihood.initialize(**hypers_likelihood)
        if init_method == "random_init_noise":
            hypers_likelihood = {'likelihood.noise_covar.noise': torch.tensor(0.1**2)}  
            likelihood.initialize(**hypers_likelihood)
        self.model = model
        self.likelihood = likelihood
        self.use_ngd = use_ngd
        self.ngd_lr = ngd_lr
        self.save_model = save_model
        self.save_path = f"./saved_models/{self.obj_name}-{self.dim}_{self.method_args['init']['model']}_m{m}_{init_method}"

        return self

    def train(self, lr=0.1, num_epochs=10, 
        scheduler="multistep", gamma=1.0, 
        train_batch_size=1024,
        mll_type="PLL", beta=1.0,
        load_run=None,
        learn_S_only=False,
        separate_group=None, lr2=None, gamma2=None,
        learn_variational_only=False, learn_hyper_only=False,
        ):

        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()

        load_run_path = self.save_path + "_" + load_run + ".model" if load_run is not None else None
        print("Loading previous run: ", load_run)

        means, variances, rmse, test_nll, testing_time = eval_gp(
            self.model, self.likelihood, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=None)
        print(f"initial test rmse: {rmse:.4e}, test nll: {test_nll:.4e}")
        
        self.model, self.likelihood, _, = train_gp(
            self.model, self.likelihood, 
            self.train_x, self.train_y, 
            num_epochs=num_epochs, 
            train_batch_size=train_batch_size,
            learn_inducing_values=self.learn_m,
            lr=lr,
            scheduler=scheduler, 
            gamma=gamma,
            elbo_beta=beta,
            mll_type=mll_type,
            device=self.device,
            tracker=self.tracker,
            use_ngd=self.use_ngd, ngd_lr=self.ngd_lr,
            save_model=self.save_model,
            save_path=self.save_path + f'_{wandb.run.name}',
            test_x=self.test_x, test_y=self.test_y,
            val_x=self.val_x, val_y=self.val_y,
            load_run_path=load_run_path,
            learn_S_only=learn_S_only,
            separate_group=separate_group, lr2=lr2, gamma2=gamma2,
            learn_variational_only=learn_variational_only,
            learn_hyper_only=learn_hyper_only,
        )

        if self.save_model:
            save_path = self.save_path + f'_{wandb.run.name}'
            state = {"model": self.model.state_dict(), "epoch": num_epochs}
            torch.save(state, f'{save_path}.model')
            print("Finish training, model saved to ", save_path)
        return self

    def eval(self, step=99999):
        means, variances, rmse, test_nll, testing_time = eval_gp(
            self.model, self.likelihood, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=self.tracker, step=step)
        return self

if __name__ == "__main__":
    fire.Fire(SVGP_exp)

# use lm initialization
# python eval_svgp.py --obj_name 3droad --dim 2 - init_hypers --num_inducing 50 --init_method lm --init_expid TEST - train --num_epochs 300 --lr 0.0005 --scheduler multistep --gamma 0.1 --train_batch_size 1024 --elbo_beta 0.1 --mll_type PLL done

# use kmeans initialization
# python eval_svgp.py --obj_name 3droad --dim 2 - init_hypers --num_inducing 50 --init_method kmeans - train --num_epochs 300 --lr 0.01 --scheduler multistep --gamma 0.1 --train_batch_size 1024 --elbo_beta 1.0 --mll_type PLL done