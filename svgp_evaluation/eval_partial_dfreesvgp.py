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
from partial_dsvgp_free import GPModel, get_inducing_points, get_inducing_directions, train_gp, eval_gp
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

class PartialDfreeSVGP_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="PartialDfreeSVGP")

    def init_hypers(self, num_inducing=2, 
        init_method="random", 
        total_num_directions=2,
        init_expid='-',
        m=0,
        learn_inducing_locations=True, 
        use_ngd=False, ngd_lr=0.1,
        save_model=False):

        self.method_args['init_hypers'] = locals()
        del self.method_args['init_hypers']['self']
        m = num_inducing + total_num_directions if m == 0 else m
        # m is the size of Kuu, recorded to maintain the same computational cost
        self.method_args['init_hypers']['m'] = m

        self.learn_inducing_locations = learn_inducing_locations

        if init_method == "random":
            rand_index = random.sample(range(self.train_n), num_inducing)
            u0 = self.train_x[rand_index, :] 
            # randomly give total_num_directions directions to some point
            rand_index_with_direction = random.sample(range(num_inducing), total_num_directions)
            inducing_directions = torch.eye(self.dim)[:1] 
            inducing_directions = inducing_directions.repeat(total_num_directions,1)
            num_directions = 1
            inducing_values_num = torch.zeros(num_inducing, dtype=int)
            inducing_values_num[rand_index_with_direction] = 1

        elif init_method == "kmeans":
            from sklearn.cluster import KMeans
            xk = self.train_x.numpy()
            kmeans = KMeans(n_clusters=num_inducing, random_state=0).fit(xk)
            u_kmeans = kmeans.cluster_centers_
            u0 = torch.tensor(u_kmeans)
            # randomly give total_num_directions directions to some point
            rand_index_with_direction = random.sample(range(num_inducing), total_num_directions)
            inducing_directions = torch.eye(self.dim)[:1] 
            inducing_directions = inducing_directions.repeat(total_num_directions,1)
            num_directions = 1
            inducing_values_num = torch.zeros(num_inducing, dtype=int)
            inducing_values_num[rand_index_with_direction] = 1

        elif init_method.startswith("lm") or init_method.startswith("tr_newton"):
            res = pkl.load(open(f"../results/{self.obj_name}-{self.dim}_{init_method}_m{m}_{init_expid}_{self.seed}.pkl", "rb"))
            u0 = res["u"]
            c = res["c"]
            sigma = res["sigma"]
            theta = res["theta"]
            inducing_directions = res["V_mat"]
            inducing_values_num = res["inducing_values_num"]
            inducing_directions = torch.tensor(inducing_directions)
            inducing_values_num = torch.tensor(inducing_values_num)
            num_directions = inducing_values_num.max().item()
            assert total_num_directions == inducing_values_num.sum().item()

        model = GPModel(inducing_points=u0, 
            inducing_directions=inducing_directions,
            inducing_values_num=inducing_values_num,
            learn_inducing_locations=learn_inducing_locations,
            use_ngd=use_ngd)

        if init_method.startswith("lm") or init_method.startswith("tr_newton"):
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(sigma).to(u0.device),
                'covar_module.lengthscale': torch.tensor(theta).to(u0.device),
                }
            if use_ngd:
                hypers["variational_strategy._variational_distribution.natural_vec"] = c.to(u0.device)
            else:
                hypers["variational_strategy._variational_distribution.variational_mean"] = c.to(u0.device)
            model.initialize(**hypers)

        self.model = model
        self.likelihood = model.likelihood
        self.num_directions = num_directions # the dummy same number of directions in the full model
        self.use_ngd = use_ngd
        self.ngd_lr = ngd_lr
        self.save_model = save_model
        self.save_path = f'./saved_models/{self.obj_name}-{self.dim}_m{m}_{init_method}'

        return self

    def train(self, lr=0.1, num_epochs=10, 
        scheduler=None, gamma=1.0, 
        train_batch_size=1024,
        mll_type="PLL", elbo_beta=0.1, 
        load_run=None):

        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()

        load_run_path = self.save_path + "_" + load_run + ".model" if load_run is not None else None
        print("Loading previous run: ", load_run)
        self.model, self.likelihood, _, = train_gp(
            self.model, self.likelihood, 
            self.train_x, self.train_y, 
            num_directions=self.num_directions,
            num_epochs=num_epochs, 
            train_batch_size=train_batch_size,
            lr=lr,
            scheduler=scheduler, 
            gamma=gamma,
            elbo_beta=elbo_beta,
            mll_type=mll_type,
            device=self.device,
            tracker=self.tracker,
            use_ngd=self.use_ngd, ngd_lr=self.ngd_lr,
            save_model=self.save_model,
            save_path=self.save_path+f'_{wandb.run.name}',
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
            num_directions=self.num_directions,
            test_batch_size=2048,
            device=self.device,
            tracker=self.tracker, step=step)
        return self

if __name__ == "__main__":
    fire.Fire(PartialDfreeSVGP_exp)
