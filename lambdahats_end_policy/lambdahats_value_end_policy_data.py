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


losses = list()

min_modelno = 0
max_modelno = 8000

if args.device == 0:
    min_modelno = 0
    max_modelno = max_modelno//4
elif args.device == 1:
    min_modelno = max_modelno//4
    max_modelno = max_modelno//2
elif args.device == 2:
    min_modelno = max_modelno//2
    max_modelno = max_modelno//4*3
elif args.device == 3:
    min_modelno = max_modelno//4*3
    max_modelno = max_modelno

os.makedirs(f"checkpoint_stats/end_policy/{min_modelno}-{max_modelno}", exist_ok=True)

dataloader, dataset, _ = lah.get_artifact_network_and_data(
    8000, 
    datapoints = 2000, 
    batch_size = 500, 
    download = False, 
    device = device, 
    shuffle = True, 
    generate_dataset = True
)

for modelno in tqdm(range(min_modelno, max_modelno)):
    _, _, value_network = lah.get_artifact_network_and_data(
        modelno, 
        datapoints = 0, 
        batch_size = 0, 
        download = False, 
        device = device, 
        shuffle = True, 
        generate_dataset = False
    )

    loss = lah.evaluate_value_network(
        value_network, 
        dataloader,
        device = device
    )

    losses.append(loss)

    with open(f"checkpoint_stats/end_policy/{min_modelno}-{max_modelno}/losses.pickle", "wb") as f:
        pickle.dump(losses, f)