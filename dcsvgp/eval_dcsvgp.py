# baseline GP regression, use all training data 
import numpy as np
import time
import sys
import random
import torch
import gpytorch
import pickle as pkl
from models.dcsvgp import GPModel, train_gp, eval_gp
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

class DCSVGP_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="DCSVGP")
        
    def init_hypers(self, num_inducing=500, 
        use_ngd=False, ngd_lr=0.1,
        save_model=False,
        learn_inducing_locations=True,
        load_u=None,
        ID=None,
        ARD=False,
        kernel_type="se",
        ):

        self.method_args['init_hypers'] = locals()
        # m is the size of Kuu, recorded to maintain the same computational cost
        m = num_inducing
        self.method_args['init_hypers']['m'] = m
        del self.method_args['init_hypers']['self']
        
        rand_index = random.sample(range(self.train_n), num_inducing)
        u0 = self.train_x[rand_index, :]

        if load_u is not None:
            u0 = pkl.load(open(f'u_{load_u}_{self.obj_name}.pkl', 'rb'))
            print(f"Loaded u from {load_u}.")
        
        ard_num_dims=self.dim if ARD else None
        u0 = torch.tensor(u0)
        model = GPModel(inducing_points=u0, 
                learn_inducing_locations=learn_inducing_locations,
                ard_num_dims=ard_num_dims,
                kernel_type=kernel_type,
                )

        self.model = model
        self.use_ngd = use_ngd
        self.ngd_lr = ngd_lr
        self.save_model = save_model
        self.save_path = f"./saved_models/whitened_{self.obj_name}_{self.method_args['init']['model']}_m{m}_ID_{ID}"

        return self

    def train(self, lr=0.1, num_epochs=10, 
        scheduler="multistep", gamma=1.0, 
        train_batch_size=1024,
        mll_type="PLL", beta=1.0,
        load_run=None,
        save_u=False,lengthscale_only=False,
        alpha=-1,
        ):

        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()

        if alpha == -1:
            alpha_type = "None" 
        elif alpha == 0.1:
            alpha_type = "01"
        else:
            alpha_type = alpha

        load_run_path = self.save_path + "_" + mll_type + f"_alpha_{alpha_type}" + "_" + load_run + ".model" if load_run is not None else None
        print("Loading previous run: ", load_run)

        means, variances, rmse, test_nll = eval_gp(
            self.model, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=None)
        print(f"initial test rmse: {rmse:.4e}, test nll: {test_nll:.4e}")
        
        self.model = train_gp(
            self.model, 
            self.train_x, self.train_y, 
            num_epochs=num_epochs, 
            train_batch_size=train_batch_size,
            lr=lr, gamma=gamma,
            elbo_beta=beta,
            mll_type=mll_type,
            device=self.device,
            tracker=self.tracker,
            save_model=self.save_model,
            save_path=self.save_path + f'_{mll_type}_alpha_{alpha_type}_{wandb.run.name}',
            load_run_path=load_run_path,
            test_x=self.test_x, test_y=self.test_y,
            val_x=self.val_x, val_y=self.val_y,
            save_u=save_u, obj_name=self.obj_name,
            lengthscale_only=lengthscale_only,
            alpha=alpha,
        )

        if self.save_model:
            save_path = self.save_path + f'_{mll_type}_alpha_{alpha_type}_{wandb.run.name}'
            state = {"model": self.model.state_dict(), "epoch": num_epochs}
            torch.save(state, f'{save_path}.model')
            print("Finish training, model saved to ", save_path)
        return self

if __name__ == "__main__":
    fire.Fire(DCSVGP_exp)
