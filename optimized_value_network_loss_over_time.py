import lambdahat_helpers as lah
import time
import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default="0")
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

os.makedirs(f"checkpoint_stats/losses/", exist_ok=True)

losses = list()

min_modelno = 0
max_modelno = 8000
if args.device == 0:
    min_modelno = 0
    max_modelno = 2000
elif args.device == 1:
    min_modelno = 2000
    max_modelno = 4000
elif args.device == 2:
    min_modelno = 4000
    max_modelno = 6000
elif args.device == 3:
    min_modelno = 6000
    max_modelno = 8000


for modelno in range(min_modelno, max_modelno):
    dataloader, dataset, value_network = lah.get_artifact_network_and_data(
        modelno, 
        datapoints = 2000, 
        batch_size = 500, 
        download = False, 
        device = device, 
        shuffle = True, 
        generate_dataset = True
    )

    value_network, loss = lah.optimize_value_network(
        value_network, 
        dataloader,
        epochs = 200, 
        device = device,
        return_loss = True
    )
    losses.append(loss)

    with open(f"checkpoint_stats/losses/{min_modelno}-{max_modelno}.pkl", "wb") as f:
        pickle.dump(losses, f)