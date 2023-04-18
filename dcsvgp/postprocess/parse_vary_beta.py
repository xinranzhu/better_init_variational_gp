import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="1D", help='name of dataset')
    parser.add_argument('--mll_type', type=str, default="ELBO", help='mll type, ELBO or PLL')
    return parser.parse_args()

def get_results_from_line(line):
    info = line.split("\t")
    tuner = info[0]
    results = list(map(lambda x: float(x.split()[0]), info[1:]))
    return tuner, results


def main(args):
    summary = {}
    my_source = f"./data_UCI_vary_beta2/vary_beta_{args.dataset_name}_{args.mll_type}.txt"
    save_path = f"./data_UCI_vary_beta2/vary_beta_{args.dataset_name}_{args.mll_type}.pkl"
    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split('\t')
            if info[0] == args.dataset_name:
                summary["dataset_name"] = info[0]
                summary["n_trials"] = int(info[1])
                assert info[2] == args.mll_type
                summary["mll_type"] = info[2]
                summary["data"] = {"Mean": [], "STD": []}
                summary["models"] = []
            elif info[0] == "Mean":
                metrics = info[1:]
                metrics[-1] = 'covar_ls'
                summary['metrics'] = metrics
                METRIC="Mean"
            elif info[0] == "STD":
                METRIC="STD"
            else:
                data_cur = [float(item) for item in info[1:]]
                if METRIC == "Mean":
                    model = info[0]
                    summary["models"].append(model)
                summary["data"][METRIC].append(data_cur)
            line = f.readline()
            


    print(summary)
    # summary = sorted(summary, key=itemgetter(4))
    # import pdb; pdb.set_trace()
    pickle.dump(summary, open(save_path, "wb"))  
    print("Data saved to, ", save_path)

if __name__ == "__main__":
    main(parse_args())
