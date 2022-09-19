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
from dsvgp_free import GPModel, get_inducing_points, get_inducing_directions, train_gp, eval_gp
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

class DfreeSVGP_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="DfreeSVGP")

    def init_hypers(self, num_inducing=2, 
        num_directions=1,
        init_method="random", 
        learn_inducing_locations=True, 
        use_ngd=False, ngd_lr=0.1,
        learn_inducing_values=True,
        save_model=False):

        self.method_args['init_hypers'] = locals()
        del self.method_args['init_hypers']['self']
        # m is the size of Kuu, recorded to maintain the same computational cost
        m = int(num_inducing*(num_directions+1))
        self.method_args['init_hypers']['m'] = m

        self.learn_inducing_locations = learn_inducing_locations

        if init_method == "random":
            rand_index = random.sample(range(self.train_n), num_inducing)
            u0 = self.train_x[rand_index, :] 
        elif init_method == "kmeans":
            from sklearn.cluster import KMeans
            xk = self.train_x.numpy()
            kmeans = KMeans(n_clusters=num_inducing, random_state=0).fit(xk)
            u_kmeans = kmeans.cluster_centers_
            u0 = torch.tensor(u_kmeans)

        inducing_directions = torch.eye(self.dim)[:num_directions] 
        inducing_directions = inducing_directions.repeat(num_inducing,1)
        #inducing_directions = torch.rand(num_inducing*num_directions,dim)
        #inducing_directions = (inducing_directions.T/torch.norm(inducing_directions,dim=1)).T

        model = GPModel(inducing_points=u0, 
            inducing_directions=inducing_directions,
            learn_inducing_locations=learn_inducing_locations,
            use_ngd=use_ngd)
            
        self.model = model
        self.likelihood = model.likelihood
        self.use_ngd = use_ngd
        self.ngd_lr = ngd_lr
        self.save_model = save_model
        self.save_path = f'./saved_models/{self.obj_name}-{self.dim}_m{m}_{init_method}'

        return self

    def train(self, lr=0.1, num_epochs=10, 
        scheduler=None, gamma=0.3, 
        train_batch_size=1024,
        mll_type="ELBO", elbo_beta=0.1):

        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()
        
        means, variances, rmse, test_nll, testing_time = eval_gp(
            self.model, self.likelihood, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=self.tracker, step=0)
            
        self.model, self.likelihood, _, = train_gp(
            self.model, self.likelihood, 
            self.train_x, self.train_y, 
            num_epochs=num_epochs, 
            train_batch_size=train_batch_size,
            lr=lr,
            scheduler=scheduler, 
            gamma=gamma,
            elbo_beta=elbo_beta,
            mll_type=mll_type,
            device=self.device,
            tracker=self.tracker,
            use_ngd=self.use_ngd, ngd_lr=self.ngd_lr)

        if self.save_model:
            save_path = self.save_path + f'{wandb.run.name}'
            torch.save(self.model.state_dict(), f'{save_path}.model')
            print("Finish training, model saved to ", save_path)
            
        return self

    def eval(self, step=99999):
        means, variances, rmse, test_nll, testing_time = eval_gp(
            self.model, self.likelihood, 
            self.test_x, self.test_y, 
            test_batch_size=1024,
            device=self.device,
            tracker=self.tracker, step=step)
        return self

if __name__ == "__main__":
    fire.Fire(DfreeSVGP_exp)
