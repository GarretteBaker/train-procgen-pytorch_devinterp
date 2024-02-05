import lambdahat_helpers as lah
import time
import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8" for the alternative configuration
# torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default="0")
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

epsilons = np.logspace(-9, -5, 16)
gammas = np.logspace(3, 7, 16)
modelno = 6000

# allocate epsilons according to the GPU number (there are 4)
if args.device == 0:
    epsilons = epsilons[:4]
elif args.device == 1:
    epsilons = epsilons[4:8]
elif args.device == 2:
    epsilons = epsilons[8:12]
elif args.device == 3:
    epsilons = epsilons[12:]

spreads = dict()
results_dict = dict()
os.makedirs(f"variance_data/spreads/{modelno}-{modelno+10}", exist_ok=True)
os.makedirs(f"variance_data/results/{modelno}-{modelno+10}", exist_ok=True)
# TODO: try keeping datapoints constant
# TODO: Plot the loss of the optimized value networks over time
# TODO: Off policy value estimation with optimal policy potentials? Probably in Sutton & Barto
for epsilon in epsilons:
    for gamma in gammas:
        spread, results = lah.measure_lambdahat_local_variance(
            modelno, 
            10, 
            epsilon, 
            gamma, 
            15, 
            100,
            device = device, 
            datapoints = 2000, # TODO: run with 20,000 datapoints
            batch_size = 500, 
            datasetnloader=None, 
            optimize=True
        )
        spreads[(epsilon, gamma)] = spread
        results_dict[(epsilon, gamma)] = results
        with open(f"variance_data/spreads/constant_data/{modelno}-{modelno+10}/{epsilons[0]}-{epsilons[-1]}.pkl", "wb") as f:
            pickle.dump(spreads, f)
        with open(f"variance_data/results/constant_data/{modelno}-{modelno+10}/{epsilons[0]}-{epsilons[-1]}.pkl", "wb") as f:
            pickle.dump(results_dict, f)