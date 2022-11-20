# baseline GP regression, use all training data 
import numpy as np
import time
import sys
import random
import torch
import gpytorch
import pickle as pkl
sys.path.append("./models")
from sgpr import GPModel, train_gp, eval_gp
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

class SGPR_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="SGPR")
        
    def init_hypers(self, num_inducing=2, 
        init_method="random", 
        init_expid="-",
        save_model=False,
        init_theta=True,
        init_noise=True,
        lm_step=None,
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
            print(f"Pretraining by {init_method} cost: {time_cost} sec.")
            assert u0.shape[0] == num_inducing and u0.shape[1] == self.dim
        
        u0 = torch.tensor(u0)
        model = GPModel(self.train_x, 
            self.train_y,
            inducing_points=u0, 
            )
        
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
            model = GPModel(self.train_x, 
                self.train_y,
                inducing_points=u0_new, 
                )

        if init_method not in {"random", "kmeans", "random_init_noise", "pivchol"}:
            hypers = {}
            if init_theta:
                print("initializing theta.")
                hypers['covar_module.lengthscale'] =  torch.tensor(theta)
            if init_noise: 
                print("initializing noise.")
                hypers["likelihood.noise_covar.noise"] = torch.tensor(sigma**2)
            model.initialize(**hypers)
        if init_method == "random_init_noise":
            hypers = {'likelihood.noise_covar.noise': torch.tensor(0.1**2)}  
            model.initialize(**hypers)

        self.model = model
        self.save_model = save_model
        self.save_path = f"./saved_models/{self.obj_name}-{self.dim}_{self.method_args['init']['model']}_m{m}_{init_method}"

        return self

    def train(self, 
        lr=0.1, 
        num_epochs=10, 
        gamma=1.0, 
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

        print(f"initial test rmse: {test_rmse:.4e}, test nll: {test_nll:.4e}")
        
        self.model, _, = train_gp(
            self.model, 
            self.train_x, self.train_y, 
            num_epochs=num_epochs, 
            lr=lr,
            gamma=gamma,
            device=self.device,
            tracker=self.tracker,
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
        means, variances, test_rmse, test_nll = eval_gp(
            self.model,
            self.test_x, self.test_y, 
            device=self.device,
            tracker=self.tracker, step=step)
        return self

if __name__ == "__main__":
    fire.Fire(SGPR_exp)

