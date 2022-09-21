import numpy as np
import torch
import sys
sys.path.append("../src")
from utils import load_data

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
                data_loader=True):

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
        if data_loader:
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = load_data(dataset=obj_name, device=self.device, seed=seed)
            self.train_n = self.train_x.shape[0]
            self.test_n = self.test_x.shape[0]
            self.val_n = self.val_x.shape[0]
        else:
            xx_data = np.loadtxt(f'../data/{obj_name}-{dim}_xx_data.csv', delimiter=",",dtype='float')
            xx_truth = np.loadtxt(f'../data/{obj_name}-{dim}_xx_truth.csv', delimiter=",",dtype='float')
            y_data = np.loadtxt(f'../data/{obj_name}-{dim}_y_data.csv', delimiter=",",dtype='float')
            y_truth = np.loadtxt(f'../data/{obj_name}-{dim}_y_truth.csv', delimiter=",",dtype='float')
            self.train_n = xx_data.shape[0]
            test_n = int(xx_truth.shape[0]*2/3)
            val_n = xx_truth.shape[0] - test_n
            self.test_n = test_n
            self.val_n = val_n

            X = np.concatenate([xx_data, xx_truth], axis=0)
            y = np.concatenate([y_data, y_truth], axis=0)
            Xy = np.concatenate([X,y.reshape(-1,1)], axis=1)
            # randomly shuffle the data
            if seed > 0:
                np.random.seed(seed)
                np.random.shuffle(Xy)

            self.train_x = torch.tensor(Xy[:self.train_n,:-1])
            self.train_y = torch.tensor(Xy[:self.train_n,-1])
            self.val_x = torch.tensor(Xy[self.train_n:,:-1])[:val_n]
            self.val_y = torch.tensor(Xy[self.train_n:,-1])[:val_n]
            self.test_x = torch.tensor(Xy[self.train_n:,:-1])[val_n:]
            self.test_y = torch.tensor(Xy[self.train_n:,-1])[val_n:]
        
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
