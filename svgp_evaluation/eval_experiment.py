import numpy as np
import torch

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
                use_gpu=True, wandb=True):

        torch.set_default_dtype(torch.double)
        self.dtype = torch.get_default_dtype()
        if torch.cuda.is_available() and use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # load training and testing data
        xx_data = np.loadtxt(f'../data/{obj_name}-{dim}_xx_data.csv', delimiter=",",dtype='float')
        xx_truth = np.loadtxt(f'../data/{obj_name}-{dim}_xx_truth.csv', delimiter=",",dtype='float')
        y_data = np.loadtxt(f'../data/{obj_name}-{dim}_y_data.csv', delimiter=",",dtype='float')
        y_truth = np.loadtxt(f'../data/{obj_name}-{dim}_y_truth.csv', delimiter=",",dtype='float')
        self.dim = dim
        self.obj_name = obj_name
        self.train_n = xx_data.shape[0]
        self.train_x = torch.tensor(xx_data)
        self.train_y = torch.tensor(y_data)
        

        # split out a validation set
        test_n = int(xx_truth.shape[0]*2/3)
        val_n = xx_truth.shape[0] - test_n
        self.test_n = test_n
        self.val_n = val_n
        self.val_x = torch.tensor(xx_truth)[:val_n]
        self.val_y = torch.tensor(y_truth)[:val_n]
        self.test_x = torch.tensor(xx_truth)[val_n:]
        self.test_y = torch.tensor(y_truth)[val_n:]
        
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
