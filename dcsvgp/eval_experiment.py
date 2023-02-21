import numpy as np
import torch
import scipy
import sys
sys.path.append("../src")
from utils import load_data, load_data_1d

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
                wandb_project_name='better_init_variational_GP2', 
                wandb_entity="xinranzhu", 
                use_gpu=True, wandb=True, seed=1234,
                gradients=False,
                ):

        torch.set_default_dtype(torch.double)
        self.dtype = torch.get_default_dtype()
        torch.manual_seed(seed)
        if torch.cuda.is_available() and use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.obj_name = obj_name
        self.seed = seed

        if gradients:
            # testfun = eval(f"{obj_name}_with_deriv")()
            # n_test = 10000
            n = 10000 
            # args ={"derivative": True, "seed":seed}
            # x, y, _ = load_synthetic_data(testfun, n+n_test, **args)
            xy = scipy.io.loadmat(f"./synthetic_data/{obj_name}.mat")
            perm = torch.randperm(n)
            x = torch.tensor(xy["x"])[perm]
            y = torch.tensor(xy["y"])[perm]
            self.train_x = x[:n, :]
            self.test_x = x[n:, :]
            self.train_y = y[:n, ...]
            self.test_y = y[n:, ...]
            self.val_x, self.val_y = None, None
            self.train_n = self.train_x.shape[0]
            self.test_n = self.test_x.shape[0]
            self.val_n = 0
        else:
            if obj_name == "1D":
                self.train_x, self.train_y, self.test_x, self.test_y = load_data_1d()
                self.val_x, self.val_y = None, None
            else:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = load_data(dataset=obj_name, seed=seed)
            self.train_n = self.train_x.shape[0]
            self.test_n = self.test_x.shape[0]
            self.val_n = self.val_x.shape[0] if self.val_x is not None else 0
        
        self.dim = self.train_x.shape[1]
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
        print("wand run: ", wandb.run.name)
        return self

    def done(self):
        return None


def new(**kwargs):
    return Experiment(**kwargs)

if __name__ == "__main__":
    fire.Fire(Experiment)
