# baseline GP regression, use all training data 
from pyexpat import model
import numpy as np
import time
import sys
import torch
import gpytorch
import pickle as pkl
sys.path.append("./gps")
from gp import ExactGPModel, train_gp, eval_gp
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

class GP_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="GP")
        model = ExactGPModel(self.train_x, self.train_y)
        self.model = model
        self.likelihood = model.likelihood

    def init_hypers(self):
        self.method_args['init_hypers'] = {'m': self.train_n}
        # don't consider smart initialization of GP hyperparmas for now
        return self

    def train(self, lr=0.1, num_epochs=10, scheduler=None, gamma=1.0):
        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()

        means, variances, rmse, test_nll, testing_time = eval_gp(
            self.model, self.likelihood, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=self.tracker, step=0)

        self.model, self.likelihood, _ = train_gp(
            self.model, self.likelihood, 
            self.train_x, self.train_y, 
            num_epochs=num_epochs, 
            lr=lr, device=self.device,
            scheduler=scheduler, 
            gamma=gamma,
            tracker=self.tracker)

        return self

    def eval(self, step=99999):
        means, variances, rmse, test_nll, testing_time = eval_gp(
            self.model, self.likelihood, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=self.tracker, step=step)
        return self

if __name__ == "__main__":
    fire.Fire(GP_exp)
