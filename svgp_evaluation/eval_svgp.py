# baseline GP regression, use all training data 
from pyexpat import model
import numpy as np
import time
import sys
import random
import torch
import gpytorch
import pickle as pkl
sys.path.append("./models")
from svgp import GPModel, get_inducing_points, train_gp, eval_gp
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
        learn_inducing_locations=True, 
        learn_inducing_values=True,
        use_ngd=False, ngd_lr=0.1,
        save_model=False):

        self.method_args['init_hypers'] = locals()
        # m is the size of Kuu, recorded to maintain the same computational cost
        m = num_inducing
        self.method_args['init_hypers']['m'] = m
        del self.method_args['init_hypers']['self']

        self.learn_inducing_locations = learn_inducing_locations
        self.learn_inducing_values = learn_inducing_values

        if init_method == "random":
            rand_index = random.sample(range(self.train_n), num_inducing)
            u0 = self.train_x[rand_index, :]
        elif init_method == "kmeans":
            from sklearn.cluster import KMeans
            xk = self.train_x.cpu().numpy()
            kmeans = KMeans(n_clusters=num_inducing, random_state=0).fit(xk)
            u0 = kmeans.cluster_centers_
        else: # method = "fwd", "lm" or "tr_newton"
            res = pkl.load(open(f"../results/{self.obj_name}-{self.dim}_{init_method}_m{m}_{init_expid}_{self.seed}.pkl", "rb"))
            u0 = res["u"]
            c = res["c"]
            sigma = res["sigma"]
            theta = res["theta"]
            time_cost = res["time"]
            print(f"Pretraining by {init_method} cost: {time_cost} sec.")
            assert u0.shape[0] == num_inducing and u0.shape[1] == self.dim

        u0 = torch.tensor(u0)
        model = GPModel(inducing_points=u0, 
            learn_inducing_locations=learn_inducing_locations,
            use_ngd=use_ngd)

        if init_method not in {"random", "kmeans"}:
            print("Initializing noise, theta and variational_mean")
            hypers = {
                    'likelihood.noise_covar.noise': torch.tensor(sigma),
                    'covar_module.lengthscale': torch.tensor(theta),
                    }
            if use_ngd:
                hypers["variational_strategy._variational_distribution.natural_vec"] = torch.tensor(c).to(u0.device)
            else:
                hypers["variational_strategy._variational_distribution.variational_mean"] = torch.tensor(c).to(u0.device)
            model.initialize(**hypers)
            
        self.model = model
        self.likelihood = model.likelihood
        self.use_ngd = use_ngd
        self.ngd_lr = ngd_lr
        self.save_model = save_model
        self.save_path = f"./saved_models/{self.obj_name}-{self.dim}_{self.method_args['init']['model']}_m{m}_{init_method}"

        return self

    def train(self, lr=0.1, num_epochs=10, 
        scheduler="multistep", gamma=1.0, 
        train_batch_size=1024,
        mll_type="PLL", elbo_beta=0.1,
        load_run=None):

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
        print("Initial test: rmse = ", rmse)
        print("learn_inducing_values = ", self.learn_inducing_values)
        print("learn_inducing_locations = ", self.learn_inducing_locations)
        
        self.model, self.likelihood, _, = train_gp(
            self.model, self.likelihood, 
            self.train_x, self.train_y, 
            num_epochs=num_epochs, 
            train_batch_size=train_batch_size,
            learn_inducing_values=self.learn_inducing_values,
            lr=lr,
            scheduler=scheduler, 
            gamma=gamma,
            elbo_beta=elbo_beta,
            mll_type=mll_type,
            device=self.device,
            tracker=self.tracker,
            use_ngd=self.use_ngd, ngd_lr=self.ngd_lr,
            save_model=self.save_model,
            save_path=self.save_path + f'_{wandb.run.name}',
            test_x=self.test_x, test_y=self.test_y,
            val_x=self.val_x, val_y=self.val_y,
            load_run_path=load_run_path)

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