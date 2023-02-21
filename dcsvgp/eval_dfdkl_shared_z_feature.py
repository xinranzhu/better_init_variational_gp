# baseline GP regression, use all training data 
import numpy as np
import time
import sys
import random
import torch
import gpytorch
import pickle as pkl
from models.dfdkl_shared_z_feature import GPModelDKL
from models.dkl import train_gp, eval_gp
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

class DFDKLSharedU_feature(Experiment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs, model="DFDKLSharedU_feature")
        
    def init_hypers(self, num_inducing=500, 
        learn_m=True,
        save_model=False,
        ID=None
        ):
        self.method_args['init_hypers'] = locals()
        # m is the size of Kuu, recorded to maintain the same computational cost
        m = num_inducing
        self.method_args['init_hypers']['m'] = m
        hidden_dims = self.dim//2
        self.method_args['init_hypers']['hidden_dims'] = hidden_dims
        del self.method_args['init_hypers']['self']

        self.learn_m = learn_m
        rand_index = random.sample(range(self.train_n), num_inducing)
        u0 = self.train_x[rand_index, :]
        u0 = torch.tensor(u0)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        model = GPModelDKL(inducing_points=u0, 
                likelihood=likelihood,
                hidden_dims=(hidden_dims, hidden_dims))
        
        self.model = model
        self.save_model = save_model
        self.save_path = f"./saved_models/{self.obj_name}_{self.method_args['init']['model']}_m{m}_ID_{ID}"

        return self

    def train(self, lr=0.1, num_epochs=10, 
        scheduler="multistep", gamma=1.0, 
        train_batch_size=1024,
        mll_type="PLL", beta=1.0,
        load_run=None,
        debug=False, verbose=True, save_u=False, lengthscale_only=False,
        ):

        self.method_args['train'] = locals()
        del self.method_args['train']['self']
        self.track_run()

        load_run_path = self.save_path + "_" + mll_type + "_" + "_alpha_None" + load_run + ".model" if load_run is not None else None
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
            lr=lr,
            scheduler=scheduler, 
            gamma=gamma,
            elbo_beta=beta,
            mll_type=mll_type,
            device=self.device,
            tracker=self.tracker,
            save_model=self.save_model,
            save_path=self.save_path + f'_{mll_type}_alpha_None_{wandb.run.name}',
            test_x=self.test_x, test_y=self.test_y,
            val_x=self.val_x, val_y=self.val_y,
            load_run_path=load_run_path,
            debug=debug, verbose=verbose,
            save_u=save_u, obj_name=self.obj_name,
            lengthscale_only=lengthscale_only,
        )

        if self.save_model:
            save_path = self.save_path + f'_{mll_type}_alpha_None_{wandb.run.name}'
            state = {"model": self.model.state_dict(), "epoch": num_epochs}
            torch.save(state, f'{save_path}.model')
            print("Finish training, model saved to ", save_path)
        return self


if __name__ == "__main__":
    fire.Fire(DFDKLSharedU_feature)

# use lm initialization
# python eval_svgp.py --obj_name 3droad --dim 2 - init_hypers --num_inducing 50 --init_method lm --init_expid TEST - train --num_epochs 300 --lr 0.0005 --scheduler multistep --gamma 0.1 --train_batch_size 1024 --elbo_beta 0.1 --mll_type PLL done

# use kmeans initialization
# python eval_svgp.py --obj_name 3droad --dim 2 - init_hypers --num_inducing 50 --init_method kmeans - train --num_epochs 300 --lr 0.01 --scheduler multistep --gamma 0.1 --train_batch_size 1024 --elbo_beta 1.0 --mll_type PLL done