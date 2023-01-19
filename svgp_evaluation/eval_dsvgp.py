# baseline GP regression, use all training data 
import numpy as np
import time
import sys
import random
import torch
import gpytorch
import pickle as pkl
sys.path.append("./models")
from dsvgp import GPModel, train_gp, eval_gp
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

class DSVGP_exp(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="DSVGP")
        
    def init_hypers(self, 
        num_inducing=500, 
        num_directions=2,
        save_model=False,
        ):

        self.num_directions = num_directions
        self.method_args['init_hypers'] = locals()
        # m is the size of Kuu, recorded to maintain the same computational cost
        m = num_inducing
        self.method_args['init_hypers']['m'] = m
        del self.method_args['init_hypers']['self']


        rand_index = random.sample(range(self.train_n), num_inducing)
        inducing_points = self.train_x[rand_index, :]
        inducing_points = torch.tensor(inducing_points)
        inducing_directions = torch.eye(self.dim)[:num_directions] # canonical directions
        inducing_directions = inducing_directions.repeat(num_inducing,1)
        model = GPModel(inducing_points=inducing_points, 
                        inducing_directions=inducing_directions)
        

        self.model = model
        self.save_model = save_model
        self.save_path = f"./saved_models/{self.obj_name}-{self.dim}_{self.method_args['init']['model']}_m{m}"

        return self

    def train(self, lr=0.1, num_epochs=10, 
        scheduler="multistep", gamma=1.0, 
        train_batch_size=1024,
        mll_type="PLL",
        load_run=None,
        debug=False, verbose=True,
        ):

        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()

        load_run_path = self.save_path + "_" + load_run + ".model" if load_run is not None else None
        print("Loading previous run: ", load_run)

        _, _, test_rmse, test_nll = eval_gp(
            self.model, 
            self.test_x, self.test_y, 
            device=self.device,
            tracker=None)
        print(f"initial test rmse: {test_rmse:.4e}, test nll: {test_nll:.4e}")
        
        self.model, _, = train_gp(
            self.model, 
            self.train_x, self.train_y, 
            num_directions=self.num_directions,
            num_epochs=num_epochs, 
            train_batch_size=train_batch_size,
            lr=lr,
            scheduler=scheduler, 
            gamma=gamma,
            mll_type=mll_type,
            device=self.device,
            tracker=self.tracker,
            save_model=self.save_model,
            save_path=self.save_path + f'_{wandb.run.name}',
            test_x=self.test_x, test_y=self.test_y,
            load_run_path=load_run_path,
            debug=debug, verbose=verbose,
        )

        if self.save_model:
            save_path = self.save_path + f'_{wandb.run.name}'
            state = {"model": self.model.state_dict(), "epoch": num_epochs}
            torch.save(state, f'{save_path}.model')
            print("Finish training, model saved to ", save_path)
        return self


if __name__ == "__main__":
    fire.Fire(DSVGP_exp)
