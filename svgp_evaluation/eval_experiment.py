import numpy as np
import torch
import sys
sys.path.append("../src")
from utils import load_data, load_data_old

try:
    import fire
except ModuleNotFoundError:
    print("Optional dependencies for experiments not installed.")

try:
    import wandb
    LOG_WANDB = True
except ModuleNotFoundError:
    LOG_WANDB = False

class Experiment(object):
    def __init__(self, obj_name="3droad", dim=1,
                model="GP",
                wandb_project_name='better_init_variational_GP-rerun-kin40k-elevator', 
                wandb_entity="xinranzhu", 
                use_gpu=True, wandb=True, seed=1234,
                ):

        torch.set_default_dtype(torch.double)
        self.dtype = torch.get_default_dtype()
        if torch.cuda.is_available() and use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.dim = dim
        self.obj_name = obj_name
        self.seed = seed

        # load training and testing data
        data_loader = 0 if obj_name in {"bike", "energy", "protein"} else 1
        if data_loader:
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = load_data(dataset=obj_name, device=self.device, seed=seed)
        else:
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = load_data_old(obj_name, dim, seed=seed)
        self.train_n = self.train_x.shape[0]
        self.test_n = self.test_x.shape[0]
        self.val_n = self.val_x.shape[0]
        
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.wandb = wandb
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        print(f"Problem: {obj_name}, dim: {dim}, train_n: {self.train_n}, test_n: {self.test_n}")     

    def track_run(self):
        if self.wandb and LOG_WANDB:
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config={k: v for method_dict in self.method_args.values() for k, v in method_dict.items()},
            )
        else:
            self.tracker = None
        return self

    def done(self):
        return None


def new(**kwargs):
    return Experiment(**kwargs)

if __name__ == "__main__":
    fire.Fire(Experiment)
