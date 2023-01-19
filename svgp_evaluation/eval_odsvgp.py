# baseline GP regression, use all training data 
import numpy as np
import time
import sys
import random
import torch
import gpytorch
import pickle as pkl
sys.path.append("./models")
from odsvgp import GPModel, train_gp, eval_gp
from pivoted import _select_inducing_points
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

class ODSVGP_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="ODSVGP")
        
    def init_hypers(self, num_inducing=2, 
        num_inducing_covar=None,
        init_method="random", 
        init_expid="-",
        save_model=False,
        lm_step=None,
        use_ngd=False,
        ngd_lr=0.1,
        ):

        self.method_args['init_hypers'] = locals()
        # m is the size of Kuu, recorded to maintain the same computational cost
        m = num_inducing
        self.method_args['init_hypers']['m'] = m
        del self.method_args['init_hypers']['self']


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
            L = res["L"]
            # get the unwhitened mean
            # c = torch.linalg.solve(L.T, c)
            # get the optimal mean 
            c = torch.matmul(L, c)
            print(f"Pretraining by {init_method} cost: {time_cost} sec.")
            assert u0.shape[0] == num_inducing and u0.shape[1] == self.dim
        
        if num_inducing_covar is not None:
            self.num_inducing_covar = num_inducing_covar
        else:
            self.num_inducing_covar = (num_inducing//7)*3
        rand_index = random.sample(range(self.train_n), self.num_inducing_covar)
        covar_inducing_points = self.train_x[rand_index, :]

        u0 = torch.tensor(u0)
        model = GPModel(mean_inducing_points=u0,
            covar_inducing_points=covar_inducing_points, 
            learn_inducing_locations=True,
            use_ngd=use_ngd)

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
                covar_inducing_points=covar_inducing_points, 
                learn_inducing_locations=True,
                use_ngd=use_ngd)

        if init_method not in {"random", "kmeans", "random_init_noise", "pivchol"}:
            print("Initializing noise, theta and variational_mean")
            hypers = {
                    'likelihood.noise_covar.noise': torch.tensor(sigma**2),
                    'covar_module2.lengthscale': torch.tensor(theta),
                    }
            if use_ngd:
                hypers["variational_strategy._variational_distribution.natural_vec"] = c.to(u0.device)
            else:
                hypers["variational_strategy._variational_distribution.variational_mean"] = c.to(u0.device)
                model.variational_strategy.variational_mean_initialized = torch.tensor(1)
            model.initialize(**hypers)
            model.variational_strategy.variational_mean_initialized = torch.tensor(1)

        self.model = model
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
        debug=False, verbose=True,
        ):

        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()

        load_run_path = self.save_path + "_" + load_run + ".model" if load_run is not None else None
        print("Loading previous run: ", load_run)

        means, variances, test_rmse, test_nll = eval_gp(
            self.model, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=None)
        print(f"initial test performance: test_rmse={test_rmse:.4f}, test_nll={test_nll:.3f}.")
        
        self.model, _, = train_gp(
            self.model, 
            self.train_x, self.train_y, 
            num_epochs=num_epochs, 
            train_batch_size=train_batch_size,
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
            debug=debug, verbose=verbose,
        )

        if self.save_model:
            save_path = self.save_path + f'_{wandb.run.name}'
            state = {"model": self.model.state_dict(), "epoch": num_epochs}
            torch.save(state, f'{save_path}.model')
            print("Finish training, model saved to ", save_path)
        return self

    def eval(self, step=99999):
        means, variances, rmse, test_nll, testing_time = eval_gp(
            self.model, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=self.tracker, step=step)
        return self

if __name__ == "__main__":
    fire.Fire(ODSVGP_exp)
