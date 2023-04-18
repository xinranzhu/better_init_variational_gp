import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basemodel', type=str, default="SVGP", help='type of metric data')
    parser.add_argument('--metric', type=str, default="RMSE_mean", help='type of metric data')
    return parser.parse_args()


def main(args):
    summary = {}
    my_source = f"./data_main_uci/{args.basemodel}_uci_{args.metric}.txt"
    save_path = f"./data_main_uci/{args.basemodel}_uci_{args.metric}.pkl"

    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split('\t')
            if info[0] in ["RMSE_mean", "NLL_mean", "RMSE_std", "NLL_std"]:
                assert info[0] == args.metric
            else:
                model = info[0]
                # import ipdb; ipdb.set_trace()
                summary[model] = [float(x) for x in info[1:]]
            line = f.readline()

    print(summary)
    # summary = sorted(summary, key=itemgetter(4))
    # import pdb; pdb.set_trace()
    pickle.dump(summary, open(save_path, "wb"))  
    print("Data saved to, ", save_path)

if __name__ == "__main__":
    main(parse_args())
